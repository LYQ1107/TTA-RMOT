"""Real SQLite leases in two CPU processes; no model, GPU or GT."""
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace


def _overlapping_requests(database,order,barrier,output):
    from flex_uda.active.teacher_client import LocalTeacher
    from flex_uda.teacher.teacher_cache import TeacherCache

    class InterleavedCache(TeacherCache):
        def get_or_claim(self,key,wait=True):
            value=super().get_or_claim(key,wait=wait)
            if key==order[0]:barrier.wait(timeout=5)
            return value

    teacher=LocalTeacher.__new__(LocalTeacher)
    teacher.cache=InterleavedCache(database)
    teacher.stats=dict(logical_calls=0,new_calls=0,cache_hits=0,forward_seconds=0.,
        equivalent_forward_seconds=0.,batches=0)
    teacher.prepare=lambda request:(request['key'],[request['key']],'prompt',
        dict(view='CROP',effective_view='CROP',clip=None,clip_fallback=False))
    teacher.backend=SimpleNamespace(forward=lambda images,prompts:[dict(yes_probability=.75,
        yes_logit=1.,no_logit=0.,batch_forward_seconds=.01*len(images),batch_size=len(images)) for _ in images])
    answers=teacher.ask_many([dict(key=key,view='CROP') for key in order])
    output.put(dict(order=order,answers=answers,stats=teacher.stats))


def test_overlapping_batches_flush_owned_leases_before_waiting(tmp_path):
    from flex_uda.teacher.teacher_cache import TeacherCache
    database=tmp_path/'cache.sqlite';TeacherCache(database).db.close()
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(2);output=ctx.Queue()
    processes=[ctx.Process(target=_overlapping_requests,args=(str(database),order,barrier,output))
        for order in (['A','B'],['B','A'])]
    for process in processes:process.start()
    for process in processes:process.join(timeout=5)
    blocked=any(process.is_alive() for process in processes)
    for process in processes:
        if process.is_alive():process.terminate()
    for process in processes:process.join(timeout=2)
    assert not blocked,'Opposite-order live leases deadlocked before either batch could run'
    assert all(process.exitcode==0 for process in processes)
    rows=[output.get(timeout=2) for _ in processes]
    assert sum(row['stats']['new_calls'] for row in rows)==2
    assert sum(row['stats']['cache_hits'] for row in rows)==2
    for row in rows:
        assert [answer['request_id'] for answer in row['answers']]==row['order']
        assert all(answer['yes_probability']==.75 for answer in row['answers'])
