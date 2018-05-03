#
# GPUStruct
#

import numpy as np
import struct
import pycuda.driver as cuda

class GPUStruct(object):
    def __init__(self, objs):
        """
        Initialize the link to the struct on the GPU device.  

        objs - must be a list of variable in the order they are in the
        C struct.  Pointers are indicated with a * as in C.
        kwargs - sets the values of this struct.

        For example, if the struct is like this:

        struct Results
        {
        unsigned int n; //, __padding;
        float k;
        float *A;
        float *B;
        };

        your initialization could look like this:

        res = GPUStruct([(np.uint32,'n', 10),
                         (np.float32,'k', 0),
                         (np.float32,'*A', np.zeros(10,dtype=np.float32)),
                         (np.float32,'*B', np.ones(10,dtype=np.float32))])

        You can then use it like this:

        func(res.get_ptr(),block=(1,1,1))

        And get data like this:

        res.copy_from_gpu()
        res.A
        res.B
        res.n

        """
        # set the objs
        #self.__formats,self.__objs = zip(*[(obj[0],obj[1]) for obj in objs])
        # make them tuples to prevent modification
        self.__objs = []
        self.__objnames = []
        inits = {}
        for obj in objs:
            oname = obj[1].replace('*','')
            self.__objs.append((obj[0],obj[1]))
            self.__objnames.append(oname)
            inits[oname] = obj[2]

        # make them both tuples
        self.__objs = tuple(self.__objs)
        self.__objnames = tuple(self.__objnames)
        #self.__objs = tuple(objs)
        #self.__objnames = tuple([obj.replace('*','') for fmt,obj in self.__objs])

        # set a dict for holding nbytes
        self.__nbytes = {}
        self.__ptrs = {}
        
        # loop over objs, setting attributes from kwargs
        for fmt,obj in self.__objs:
            if obj.find('*') == 0:
                # set the obj name without the *
                obj = obj[1:]
                # it's a pointer
                self.__ptrs[obj] = None

            # also save the data
            #setattr(self,obj,kwargs[obj])
            setattr(self,obj,inits[obj])
            
        self.__ptr = None
        self.__fromstr = None

    def __del__(self):
        # loop and delete non-none pointers
        for ptr in self.__ptrs:
            if not self.__ptrs[ptr] is None:
                # free it
                self.__ptrs[ptr].free()
                self.__ptrs[ptr] = None

        if not self.__ptr is None:
            # free the main pointer struct
            self.__ptr.free()
            self.__ptr = None

    def __str__(self):
        ostring = ""
        for oname in self.__objnames:
            ostring+="%s: %s\n" % (oname, str(getattr(self,oname)))
        return ostring
    
    def copy_to_gpu(self, skip=None):
        # get skip list
        if skip is None:
            skip = []
        
        # loop over obj and send the data for the pointers
        for fmt,obj in self.__objs:
            if obj.find('*') == 0:
                # set the obj name without the *
                obj = obj[1:]
                # verify the nbytes did not change, if so, free old
                # ptr and allocate for new one.
                # get the current bytes
                dat = np.ascontiguousarray(fmt(getattr(self,obj)))
                cur_nbytes = dat.nbytes
                if self.__nbytes.has_key(obj) and \
                       self.__nbytes[obj] != cur_nbytes:
                    # free it
                    self.__ptrs[obj].free()
                    self.__ptrs[obj] = None

                # see if we need to reallocate
                if self.__ptrs[obj] is None:
                    # create mem for the pointer
                    self.__nbytes[obj] = cur_nbytes
                    self.__ptrs[obj] = cuda.mem_alloc(cur_nbytes)

                # send the data to the memory space
                if not obj in skip:
                    cuda.memcpy_htod(self.__ptrs[obj],dat)

        # pack everything and send struct to device
        self.__packstr = self._pack()
        if self.__ptr is None:
            # send it for the first time
            self.__ptr = cuda.to_device(self.__packstr)
        else:
            # copy out to the existing pointer
            cuda.memcpy_htod(self.__ptr, self.__packstr)

        # create a fromstring to get data back
        self.__fromstr = np.array(' '*len(self.__packstr))
        
    def get_ptr(self):
        if self.__ptr is None:
            raise RuntimeError("You never called copy_to_gpu.")
        return self.__ptr

    def get_packed(self):
        return self.__packstr

    def _pack(self):
        packed = ''
        self.__fmt = ''
        topack = []
        for fmt,obj in self.__objs:
            if obj.find('*') == 0:
                # set the obj name without the *
                obj = obj[1:]
                # is pointer
                self.__fmt += 'P'
                topack.append(np.intp(int(self.__ptrs[obj])))
            else:
                # is normal, so just get it
                toadd = fmt(getattr(self,obj))
                self.__fmt += toadd.dtype.char
                topack.append(toadd)
        # pack it up
        return struct.pack(self.__fmt,*topack)

    def copy_from_gpu(self, skip=None):
        #         try:
        #             # try and get the passed struct back
        #             cuda.memcpy_dtoh(self.__fromstr, self.__ptr)
        #             self.__unpacked = struct.unpack(self.__fmt, self.__fromstr)
        #         except:
        #             # just use the original packstr
        #             self.__unpacked = struct.unpack(self.__fmt, self.__packstr)

        # get skip list
        if skip is None:
            skip = []

        # makre sure we've sent there
        if self.__fromstr is None:
            raise RuntimeError("You never called copy_to_gpu.")
        
        # try and get the passed struct back
        cuda.memcpy_dtoh(self.__fromstr, self.__ptr)
        self.__unpacked = struct.unpack(self.__fmt, self.__fromstr)

        # now fill the attributes from the unpacked data
        for ind,(fmt,obj) in enumerate(self.__objs):
            if obj.find('*') == 0:
                # set the obj name without the *
                obj = obj[1:]
                # is a pointer, so retrieve from card
                if not obj in skip:
                    # first make sure dest is correct datatype
                    setattr(self,obj,fmt(getattr(self, obj)))
                    cuda.memcpy_dtoh(getattr(self, obj),
                                     self.__ptrs[obj])
            else:
                # get it from the unpacked values
                # trying to keep the dtype with a hack
                #setattr(self, obj,
                #        getattr(np,str(getattr(self,obj).dtype))(self.__unpacked[ind]))
                setattr(self, obj,
                        fmt(self.__unpacked[ind]))
                
#     def __getattr__(self, attr):

#         if attr in self.__objnames:
#             if self.__unpacked is None:
#                 # must retrieve first
#                 self.retrieve()
#             # get the index
#             ind = self.__objnames.index(attr)
#             if '*'+attr == self.__objs[ind]:
#                 # is pointer, so retrieve from card
#                 data = getattr(self, self.__objnames[ind]+'_data')
#                 cuda.memcpy_dtoh(data,getattr(self,self.__objnames[ind]))
#                 return data
#                 #return cuda.from_device(getattr(self,self.__objnames[ind]),
#                 #                        data.shape,
#                 #                        data.dtype)
#             else:
#                 # just lookup in unpacked
#                 return self.__unpacked[ind]
#         else:
#             raise AttributeError("Attribute not found %s." % (attr))
