from dataclasses import dataclass,asdict


@dataclass
class BudgetMeter:
    actual_model_forward_batches: int=0
    actual_candidate_prompt_forwards: int=0
    actual_images: int=0
    actual_visual_tokens: int=0
    actual_language_tokens: int=0
    actual_forward_seconds: float=0.
    cache_hits: int=0
    peak_allocated_bytes: int=0

    def add(self,outputs):
        if not outputs:return
        self.actual_model_forward_batches+=1;self.actual_candidate_prompt_forwards+=len(outputs)
        self.actual_forward_seconds+=outputs[0]['batch_forward_seconds']
        for r in outputs:
            self.actual_images+=r['images'];self.actual_visual_tokens+=r['visual_tokens']
            self.actual_language_tokens+=r['language_tokens']
            self.peak_allocated_bytes=max(self.peak_allocated_bytes,r['allocated_peak_bytes'])

    def record(self):return asdict(self)
