"""Small optional CUDA Driver API adapter; importing it never initializes CUDA.

Signatures/constants follow NVIDIA CUDA 12.8 cuda.h. No dependency on torch,
CuPy, a runtime compiler, or F2Asm internals. Not hardware-tested on the Mac.
"""
import ctypes as C
import sys


class Driver:
    def __init__(self,ordinal=0):
        if sys.platform!="linux": raise RuntimeError("GPU execution requires a Linux B200 host; use --dry-run on the Mac")
        self.lib=C.CDLL("libcuda.so.1")
        self.allocations=[]; self.modules=[]; self.context=None; self.events=[]
        P=C.c_void_p; U=C.c_uint; I=C.c_int; Q=C.c_uint64; Z=C.c_size_t
        signatures={
            "cuInit":[U],"cuDriverGetVersion":[C.POINTER(I)],"cuDeviceGet":[C.POINTER(I),I],
            "cuDeviceGetName":[P,I,I],"cuDeviceGetAttribute":[C.POINTER(I),I,I],
            "cuDeviceGetPCIBusId":[P,I,I],
            "cuCtxCreate_v2":[C.POINTER(P),U,I],"cuCtxDestroy_v2":[P],"cuCtxSynchronize":[],
            "cuMemAlloc_v2":[C.POINTER(Q),Z],"cuMemFree_v2":[Q],
            "cuMemcpyHtoD_v2":[Q,P,Z],"cuMemcpyDtoH_v2":[P,Q,Z],
            "cuModuleLoadData":[C.POINTER(P),P],"cuModuleUnload":[P],
            "cuModuleGetFunction":[C.POINTER(P),P,C.c_char_p],
            "cuLaunchKernel":[P,U,U,U,U,U,U,U,P,C.POINTER(P),C.POINTER(P)],
            "cuEventCreate":[C.POINTER(P),U],"cuEventRecord":[P,P],"cuEventSynchronize":[P],
            "cuEventElapsedTime":[C.POINTER(C.c_float),P,P],"cuEventDestroy_v2":[P],
            "cuTensorMapEncodeTiled":[P,I,U,P,C.POINTER(Q),C.POINTER(Q),C.POINTER(U),C.POINTER(U),I,I,I,I]}
        for name,args in signatures.items():
            fn=getattr(self.lib,name); fn.restype=I; fn.argtypes=args
        self.call("cuInit",0)
        device=I(); self.call("cuDeviceGet",C.byref(device),ordinal); self.device=device.value
        name=C.create_string_buffer(256); self.call("cuDeviceGetName",name,256,self.device)
        pci=C.create_string_buffer(32);self.call("cuDeviceGetPCIBusId",pci,32,self.device)
        version=I(); self.call("cuDriverGetVersion",C.byref(version))
        self.info={"name":name.value.decode(),"ordinal":ordinal,"driver_version":version.value,
                   "pci_bus_id":pci.value.decode(),
                   "compute_capability":[self.attribute(75),self.attribute(76)],"sm_count":self.attribute(16),"l2_bytes":self.attribute(38)}
        if self.info["compute_capability"]!=[10,0] or "B200" not in self.info["name"]:
            raise RuntimeError(f"Suite is qualified only for B200/sm_100a, found {self.info}")
        context=P(); self.call("cuCtxCreate_v2",C.byref(context),0,self.device); self.context=context

    def call(self,name,*args):
        code=getattr(self.lib,name)(*args)
        if code: raise RuntimeError(f"{name}: CUDA error {code}")

    def attribute(self,number):
        value=C.c_int(); self.call("cuDeviceGetAttribute",C.byref(value),number,self.device); return value.value

    def upload(self,data):
        if not data: raise ValueError("Empty allocation")
        p=C.c_uint64(); self.call("cuMemAlloc_v2",C.byref(p),len(data)); self.allocations.append(p)
        buf=C.create_string_buffer(data); self.call("cuMemcpyHtoD_v2",p,buf,len(data)); return p.value

    def download(self,p,size):
        b=C.create_string_buffer(size); self.call("cuMemcpyDtoH_v2",b,p,size); return b.raw

    def load(self,binary):
        image=C.create_string_buffer(binary); module=C.c_void_p()
        self.call("cuModuleLoadData",C.byref(module),image); self.modules.append(module)
        function=C.c_void_p(); self.call("cuModuleGetFunction",C.byref(function),module,b"bench"); return function

    def tensor_map(self,pointer,shape,tiles,element_bytes):
        if element_bytes not in (4,8) or not 1<=len(shape)<=3 or pointer%16: raise ValueError("Invalid tensor map")
        dims=[shape[0]*tiles]+list(shape[1:]); strides=[]; stride=dims[0]*element_bytes
        for dimension in dims[1:]: strides.append(stride); stride*=dimension
        if any(s%16 for s in strides): raise ValueError("Tensor-map byte stride is not 16-byte aligned")
        # CUtensorMap requires 64-byte host alignment and occupies 128 bytes.
        storage=C.create_string_buffer(191); address=(C.addressof(storage)+63)&~63
        rank=len(shape)
        self.call("cuTensorMapEncodeTiled",address,2 if element_bytes==4 else 4,rank,pointer,
                  (C.c_uint64*rank)(*dims),(C.c_uint64*max(1,rank-1))(*strides),
                  (C.c_uint*rank)(*shape),(C.c_uint*rank)(*[1]*rank),0,0,0,0)
        return self.upload(C.string_at(address,128))

    def launch(self,function,blocks,threads,input_pointer,output_pointer,repetitions,count,tensor_map=0):
        args=[C.c_uint64(input_pointer),C.c_uint64(output_pointer),C.c_uint(repetitions),C.c_uint(count),C.c_uint64(tensor_map)]
        pointers=(C.c_void_p*len(args))(*[C.addressof(x) for x in args])
        self.call("cuLaunchKernel",function,blocks,1,1,threads,1,1,0,None,pointers,None)

    def event(self):
        event=C.c_void_p(); self.call("cuEventCreate",C.byref(event),0); self.events.append(event); return event

    def timed(self,callback):
        begin,end=self.event(),self.event(); self.call("cuEventRecord",begin,None)
        callback(); self.call("cuEventRecord",end,None); self.call("cuEventSynchronize",end)
        elapsed=C.c_float(); self.call("cuEventElapsedTime",C.byref(elapsed),begin,end)
        for e in (begin,end): self.call("cuEventDestroy_v2",e); self.events.remove(e)
        return elapsed.value

    def close(self):
        # After a CUDA trap/context error, cleanup may also fail; preserve the
        # original error rather than masking it with a secondary destructor error.
        for name,items in (("cuEventDestroy_v2",self.events),("cuModuleUnload",self.modules),("cuMemFree_v2",self.allocations)):
            for item in reversed(items):
                try:self.call(name,item)
                except RuntimeError:pass
        if self.context:
            try:self.call("cuCtxDestroy_v2",self.context)
            except RuntimeError:pass
            self.context=None
