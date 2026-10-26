"""Linux Landlock read whitelist plus independent Python target-GT access audit."""
import ctypes
import os
from pathlib import Path
import sys


def restrict_filesystem(read_roots,write_roots):
    libc=ctypes.CDLL(None,use_errno=True)
    abi=libc.syscall(444,0,0,1)
    if abi<1:raise RuntimeError('Landlock unavailable: do not silently downgrade to audit-only GT separation')
    handled=(1<<13)-1
    if abi>=2:handled|=1<<13
    if abi>=3:handled|=1<<14
    class Ruleset(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
    class Rule(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
    ruleset=Ruleset(handled);fd=libc.syscall(444,ctypes.byref(ruleset),ctypes.sizeof(ruleset),0)
    if fd<0:raise OSError(ctypes.get_errno(),'create Landlock ruleset')
    permissions={str(Path(p).resolve()):(1|4|8) for p in read_roots if Path(p).exists()}
    permissions.update({str(Path(p).resolve()):handled for p in write_roots if Path(p).exists()})
    for path,access in permissions.items():
        opened=os.open(path,os.O_PATH|os.O_CLOEXEC)
        try:
            if not Path(path).is_dir():access&=1|2|4
            rule=Rule(access,opened)
            if libc.syscall(445,fd,1,ctypes.byref(rule),0)!=0:raise OSError(ctypes.get_errno(),'add Landlock path '+path)
        finally:os.close(opened)
    if libc.prctl(38,1,0,0,0)!=0 or libc.syscall(446,fd,0)!=0:raise OSError(ctypes.get_errno(),'enforce Landlock')
    os.close(fd)
    return dict(landlock_abi=abi,kernel_read_whitelist=True,kernel_stage_write_denied=True,
        read_roots=list(permissions),handled_filesystem_access=handled)


def install_access_audit():
    result=dict(target_GT_open_attempts=[],target_GT_reads=0)
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]))
        if path.name in ('labels.json','gt.txt') or 'gt_template' in path.parts or 'gt_templates' in path.parts:
            result['target_GT_open_attempts'].append(str(path))
            raise PermissionError('teacher process may not open annotation payloads')
    sys.addaudithook(audit)
    return result
