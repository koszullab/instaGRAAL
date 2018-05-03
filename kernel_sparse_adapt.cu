#include <curand_kernel.h>
#define BLOCK_SIZE 512
#define EL_PER_THREAD 1
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300
#else
__device__ inline double __shfl_down(double var, unsigned int srcLane, int width=32) {
      int2 a = *reinterpret_cast<int2*>(&var);
      a.x = __shfl_down(a.x, srcLane, width);
      a.y = __shfl_down(a.y, srcLane, width);
      return *reinterpret_cast<double*>(&a);
    }
#endif

extern "C"
{


    texture<unsigned char, 2> tex;

//    texture<float, 2, cudaReadModeElementType> texData;

    typedef struct frag {
        int* pos;
        int* sub_pos; // position ( sub frags)
        int* id_c;
        int* start_bp;
        int* len_bp;
        int* sub_len; // length(sub frags)
        int* circ;
        int* id;
        int* prev;
        int* next;
        int* l_cont;
        int* sub_l_cont; // length(sub frags)
        int* l_cont_bp;
        int* ori;
        int* rep;
        int* activ;
        int* id_d;
    } frag;



//    typedef struct double14{
//        double x0;
//        double x1;
//        double x2;
//        double x3;
//        double x4;
//        double x5;
//        double x6;
//        double x7;
//        double x8;
//        double x9;
//        double x10;
//        double x11;
//        double x12;
//        double x13;
//        double x14;
//    } double14;

//    typedef struct bigfrag {
//        int* pos;
//        int* id_c;
//        int* start_bp;
//        int* circ;
//        int* l_cont_bp;
//        int* ori;
//        int* rep;
//        int* activ;
//    } bigfrag;

    typedef struct __attribute__ ((packed)) param_simu {
        float kuhn __attribute__ ((packed));
        float lm __attribute__ ((packed));
        float c1 __attribute__ ((packed));
        float slope __attribute__ ((packed));
        float d __attribute__ ((packed));
        float d_max __attribute__ ((packed));
        float fact __attribute__ ((packed));
        float v_inter __attribute__ ((packed));
    } param_simu;


    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                            __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
    __global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
    {
            int id = blockIdx.x*blockDim.x + threadIdx.x;

            if (id >= nthreads)
                    return;
            curand_init(seed, id, offset, &s[id]);
    }

    __device__ float factorial(float n)
    {

        float result = 1;
        n = floor(n);
        if (n<10){
            for(int c = 1 ; c <= n ; c++ )
                result = result * c;
        }
        else{
            result = powf(n,n) * exp(-n) * sqrtf(2 * M_PI * n) * (1 + (1/(12 * n)));
        }
        return ( result );
    }

     __device__ param_simu modify_param_simu(int id_modifier, param_simu p, float var)
     {
         param_simu out;
         if (id_modifier == 0){
            out.kuhn = p.kuhn;
            out.lm = p.lm;
            out.slope = p.slope;
            out.d = var;
            out.c1 = p.c1;
            out.d_max = p.d_max;
            out.v_inter = p.v_inter;
            out.fact = p.fact;
         }
         else if (id_modifier == 1){
            out.kuhn = p.kuhn;
            out.lm = p.lm;
            out.slope = var;
            out.d = p.d;
            out.c1 = p.c1;
            out.d_max = p.d_max;
            out.v_inter = p.v_inter;
            out.fact = p.fact;
         }
         return (out);
     }


    __device__ float rippe_contacts(float s, param_simu p)
    {
        // s = distance in kb
        // p = model's parameters
        float result = 0.0f;
        if ((s>0.0f) && (s<p.d_max)){
            result = (p.c1 * pow(s, p.slope) * exp((p.d-2)/(pow(s*p.lm/p.kuhn, 2.0f ) + p.d)  )) * p.fact;
        }
        float out = max(result, p.v_inter);
        return ( out );
    }

//    __device__ float rippe_contacts_circ(float s, float s_tot, param_simu p)
//    {
//        // s = distance in kb
//        // p = model's parameters
//        // s_tot = total length of circular contig
//        float result = 0.0f;
//        float n_dist = 1.0f;
//        float n_tot = 1.0f;
//        float n = 1.0f;
//        float K = 1.0f;
//        float norm_circ, norm_lin, nmax, val;
//        if ((s > 0.0f) && (s < p.d_max)){
////        if ((s < s_tot) && (s > 0.0) && (s < p.d_max)){
//            K = p.lm / p.kuhn;
//            n_dist = s ;
//            n_tot = s_tot;
//            nmax = K * 1;
//
//            n = K * n_dist *(n_tot - n_dist) / n_tot;
//
//            norm_lin = rippe_contacts(s, p);
//            norm_circ = (powf(p.kuhn, -3.0f) * powf(nmax, p.slope) * expf((p.d - 2.0f)/(powf(nmax, 2.0f ) + p.d))) * p.fact;
//
//            val = (powf(p.kuhn, -3.0f) * powf(n, p.slope) * expf((p.d - 2.0f)/(powf(n, 2.0f ) + p.d))) * p.fact;
//            result = val * norm_lin / norm_circ;
//            result = val;
//        }
//        float out = max(result, p.v_inter);
////        else{
////            result = p.v_inter;
////        }
//        return ( out );
//    }


    __device__ float rippe_contacts_circ(float s, float s_tot, param_simu p)
    {
        // s = distance in kb
        // p = model's parameters
        // s_tot = total length of circular contig
        float result = 0.0f;
        float n_dist = 1.0f;
        float n_tot = 1.0f;
        float n = 1.0f;
        float K = 1.0f;
        if ((s > 0.0f) && (s < p.d_max)){
//        if ((s < s_tot) && (s > 0.0) && (s < p.d_max)){
            K = p.lm / p.kuhn;
            n_dist = s ;
            n_tot = s_tot;
            n = K * n_dist *(n_tot - n_dist) / n_tot;

            result = (powf(p.kuhn, -3.0f) * powf(n, p.slope) * expf((p.d - 2.0f)/(powf(n, 2.0f ) + p.d))) * p.fact;
        }
        float out = max(result, p.d_max);
//        else{
//            result = p.v_inter;
//        }
        return ( out );

    }


    __device__ float evaluate_likelihood_pxl_float(float ex, float ob)
    {
    // ex = expected n contacts
    // ob = observed n contacts
        float res = 0.0;
        float lim = 15.0;
        if (ex != 0.0){
            if (ob >=lim){
               res = ob * log10(ex) - ex - (ob * log10(ob) - ob + log10(sqrtf(ob * 2.0f * M_PI)));
            }
            else if ((ob>0) && (ob<lim)){
                res = ob * log10(ex) - ex - log10(factorial(ob));
             }
            else if (ob==0){
                res = - ex;
            }
        }

        return (res);
    }



    __device__ double evaluate_likelihood_pxl_double(double ex, double ob)
    {
    // ex = expected n contacts
    // ob = observed n contacts
        double res = 0;
        double lim = 15;
        if (ex != 0){
            if (ob >=lim){
               res = ob * log10(ex) - ex - (ob * log10(ob) - ob + log10(sqrt(ob * 2.0 * M_PI)));
            }
            else if ((ob>0) && (ob<lim)){
                res = ob * log10(ex) - ex - log10((double) factorial((float) ob));
             }
            else if (ob==0){
                res = - ex;
            }
        }

        return (res);
    }


    __device__ int2 lin_2_2dpos(int ind)
    {
        int i = ind + 1;
        int x = (-0.5 + 0.5 * sqrt((float) 1 + 8 * (i - 1))) + 2;
        int y =  x * (3 - x) / 2 + i - 1;
        //int2 out = (int2) (x - 1,y - 1);
        int2 out;
        out.x = min(x - 1, y -1);
        out.y = max(x - 1, y - 1);
        return (out);
    }

    __device__ int conv_plan_pos_2_lin(int2 pos)
    {
        int x = pos.x + 1;
        int y = pos.y + 1;
        int i = min(x,y);
        int j = max(x,y);
        int ind = (j * (j - 3)) / 2 + i;
//        int ind = (j * (j - 3) / 2 + i + 1) - 1;
        return ind;
    }



//
//    __global__ void select_them(const int* __restrict__ spData_row,
//                                const int* __restrict__ spData_col,
//                                const int* __restrict__ vect_id_c,
//                                int *selec,
//                                int id_ctg1,
//                                int id_ctg2,
//                                int *counter,
//                                int size_arr)
//    {
//        __shared__ int selec_smem[512];
//        __shared__ int counter_smem;
//        int *counter_smem_ptr;
//        int idx = blockIdx.x * blockDim.x + threadIdx.x;
//        int condition = idx < size_arr;
//        int curr_id_ctg1, curr_id_ctg2, fi, fj;
//        int local_count = 0;
//        if ((threadIdx.x == 0) && (condition ==1))
//        {
//            counter_smem_ptr = &counter_smem;
//            counter_smem = 0;
//        }
//
//        selec_smem[threadIdx.x] = -1;
//
//        __syncthreads();
//
//       // each counting thread writes its index to shared memory
//
//        if (condition == 1){
//            fi = spData_row[idx];
//            fj = spData_col[idx];
//            curr_id_ctg1 = vect_id_c[fi];
//           // each counting thread writes its index to shared memory //
//            if ((curr_id_ctg1 == id_ctg1) || (curr_id_ctg1 == id_ctg2)) {
//                curr_id_ctg2 = vect_id_c[fj];
//                if ((curr_id_ctg2 == id_ctg1) || (curr_id_ctg2 == id_ctg2)) {
//                    local_count = atomicAdd(counter_smem_ptr, 1);
//                    selec_smem[local_count] =  idx;
//                }
//            }
//        }
//
//        __syncthreads();
//
//        if (threadIdx.x == 0)
//            counter_smem = atomicAdd(counter, counter_smem);
//
//        __syncthreads();
//
//
//        if ((selec_smem[threadIdx.x] >= 0) && (condition ==1))
//            selec[counter_smem + threadIdx.x] = selec_smem[threadIdx.x];
//
//    }




    __global__ void select_uniq_id_c(frag* fragArray,
                                     int* list_uniq_id_c,
                                     int* list_uniq_len,
                                     int* counter,
                                     int n_frags)
    {
        __shared__ int selec_smem[1024]; //2 * 512
        __shared__ int counter_smem;
        int *counter_smem_ptr;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int condition = idx < n_frags;
        int local_count = 0;
        int pos, len, id_c;
        if ((threadIdx.x == 0) && (condition ==1))
        {
            counter_smem_ptr = &counter_smem;
            counter_smem = 0;
        }

        selec_smem[threadIdx.x * 2] = -1;

        __syncthreads();

       // each counting thread writes its index to shared memory

        if (condition == 1){
            pos = fragArray->pos[idx];
           // each counting thread writes its index to shared memory //
            if (pos == 0) {
                len = fragArray->l_cont[idx];
                id_c = fragArray->id_c[idx];
                local_count = atomicAdd(counter_smem_ptr, 1);
                selec_smem[local_count * 2] =  id_c;
                selec_smem[local_count * 2 + 1] =  len;
            }
        }

        __syncthreads();

        if ((threadIdx.x == 0) && (condition ==1))
            counter_smem = atomicAdd(counter, counter_smem); // counter_smem receive the old value of counter while counte += counter_smem

        __syncthreads();


        if ((selec_smem[threadIdx.x * 2] >= 0) && (condition ==1)){
            list_uniq_id_c[counter_smem + threadIdx.x] = selec_smem[threadIdx.x * 2];
            list_uniq_len[counter_smem + threadIdx.x] = selec_smem[threadIdx.x * 2 + 1];
        }
    }


    __global__ void explode_genome(frag* fragArray,
                                   int* shuffle_order,
                                   int n_frags)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int condition = idx < n_frags;
        if (condition){
            fragArray->pos[idx] = 0;
            fragArray->start_bp[idx] = 0;
            fragArray->sub_pos[idx] = 0;
            fragArray->id_c[idx] = shuffle_order[idx];
            fragArray->prev[idx] = -1;
            fragArray->next[idx] = -1;
            fragArray->l_cont[idx] = 1;
            fragArray->l_cont_bp[idx] = fragArray->len_bp[idx];
            fragArray->sub_l_cont[idx] = fragArray->sub_len[idx];
        }
    }


    __global__ void count_num(int* list_vals,
                              int value,
                              int* counter,
                              int n_values)
    {
//        __shared__ int selec_smem[1024]; //2 * 512
        __shared__ int selec_smem[512]; //2 * 512
        __shared__ int counter_smem;
        int *counter_smem_ptr;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int condition = idx < n_values;
        int local_count = 0;
        int val;
        if ((threadIdx.x == 0) && (condition ==1))
        {
            counter_smem_ptr = &counter_smem;
            counter_smem = 0;
        }

//        selec_smem[threadIdx.x * 2] = -1;
        selec_smem[threadIdx.x] = -1;

        __syncthreads();

       // each counting thread writes its index to shared memory

        if (condition == 1){
            val = list_vals[idx];
           // each counting thread writes its index to shared memory //
            if (val == value) {
                local_count = atomicAdd(counter_smem_ptr, 1);
            }
        }
        __syncthreads();
        if ((threadIdx.x == 0) && (condition ==1)){
            counter_smem = atomicAdd(counter, counter_smem);
        }
    }



    __global__ void make_old_2_new_id_c(int* list_uniq_id_c,
                                        int* list_old_2_new_id_c,
                                        int n_contigs)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int condition = idx < n_contigs;
        int id_c;
        if (condition ==1){
            id_c = list_uniq_id_c[idx];
            list_old_2_new_id_c[id_c] = idx;
        }
    }


    __global__ void slice_sp_mat(const int* __restrict__ spData_dat,
                                 const int* __restrict__ spData_row,
                                 const int* __restrict__ spData_col,
                                 frag* fragArray,
                                 const int* __restrict__ vect_id_c,
                                 const int* __restrict__ vect_pos,
                                 int* sub_spData_row,
                                 int* sub_spData_col,
                                 int* sub_spData_dat,
                                 int id_ctg1,
                                 int id_ctg2,
                                 int id_frag_a,
                                 int id_frag_b,
                                 int n_bounds,
                                 int *counter,
                                 int size_arr)
    {

//        __shared__ int selec_smem[1536]; //512 * 3
//        __shared__ int selec_smem[768]; // 256 * 3
        __shared__ int selec_smem[SIZE_BLOCK_4_SUB_3]; // 128 * 3
        __shared__ int counter_smem;
        __shared__ int pos_fa;
        __shared__ int pos_fa_down;
        __shared__ int pos_fb;
        __shared__ int up_bound_fa;
        __shared__ int down_bound_fa;
        __shared__ int up_bound_fb;
        __shared__ int down_bound_fb;
        __shared__ int l_ctg_fa;
        __shared__ int l_ctg_fb;
        __shared__ int is_circ;

        int *counter_smem_ptr;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int curr_id_ctg1, curr_id_ctg2, fi, fj, dat;
        int tid3 = threadIdx.x * 3;
        int local_count = 0;
        int condition = idx < size_arr;
        int id_alter_contig, condition_1, condition_2, condition_3, condition_4, pos_fi, pos_fj, pos_x, pos_y;
        int same_contigs = id_ctg1 == id_ctg2;
        if ((threadIdx.x == 0))
        {
            counter_smem_ptr = &counter_smem;
            counter_smem = 0;
            int tmp_pos_fa = fragArray->sub_pos[id_frag_a];
            int tmp_pos_fb = fragArray->sub_pos[id_frag_b];
            int ori_fa = fragArray->ori[id_frag_a];
            int ori_fb = fragArray->ori[id_frag_b];
            int sub_len_fa = fragArray->sub_len[id_frag_a];
            int sub_len_fb = fragArray->sub_len[id_frag_b];
            pos_fa = max(0, tmp_pos_fa * (ori_fa == 1) + (tmp_pos_fa - sub_len_fa) * (ori_fa == -1));
            pos_fb = max(0, tmp_pos_fb * (ori_fb == 1) + (tmp_pos_fb - sub_len_fb) * (ori_fb == -1));

            is_circ = fragArray->circ[id_frag_a];

            l_ctg_fa = fragArray->sub_l_cont[id_frag_a];
            l_ctg_fb = fragArray->sub_l_cont[id_frag_b];

            up_bound_fa = max(0, pos_fa - n_bounds - sub_len_fa);
            down_bound_fa = min(l_ctg_fa - 1, pos_fa + n_bounds + sub_len_fa);

            up_bound_fb = max(0, pos_fb - sub_len_fb);
            down_bound_fb = min(l_ctg_fb - 1, pos_fb + sub_len_fb);


        }

        selec_smem[tid3] = -1;

        __syncthreads();

        if (condition == 1){
            fi = spData_row[idx];
            fj = spData_col[idx];
            dat = spData_dat[idx];
            curr_id_ctg1 = vect_id_c[fi];
           // each counting thread writes its index to shared memory //
            if ((curr_id_ctg1 == id_ctg1) || (curr_id_ctg1 == id_ctg2)) {
                curr_id_ctg2 = vect_id_c[fj];
                if ((curr_id_ctg2 == curr_id_ctg1) && (same_contigs == 1) && (is_circ == 0)){ // same_contig : smart slicing !
                    pos_fi = vect_pos[fi];
                    pos_fj = vect_pos[fj];
                    pos_x = min(pos_fi, pos_fj);
                    pos_y = max(pos_fi, pos_fj);

                    if (pos_fb > pos_fa){
                        condition_1 = (pos_x <= down_bound_fa) && (pos_y >= up_bound_fa);
                        condition_2 = (pos_y >= up_bound_fb) && (pos_x <= down_bound_fb);
                    }
                    else{
                        condition_1 = (pos_x <= down_bound_fb) && (pos_y >= up_bound_fb);
                        condition_2 = (pos_y >= up_bound_fa) && (pos_x <= down_bound_fa);
                    }

                    if ((condition_1==1) || (condition_2 == 1)){
                        local_count = atomicAdd(counter_smem_ptr, 1);
                        selec_smem[local_count * 3] =  dat;
                        selec_smem[local_count * 3 + 1] = fi;
                        selec_smem[local_count * 3 + 2] = fj;
                    }
                }
                else if ((same_contigs == 0) && (curr_id_ctg2 == id_ctg1) || (curr_id_ctg2 == id_ctg2)) { // two different contigs : retrive data of the two contigs
                    local_count = atomicAdd(counter_smem_ptr, 1);
                    selec_smem[local_count * 3] =  dat;
                    selec_smem[local_count * 3 + 1] = fi;
                    selec_smem[local_count * 3 + 2] = fj;
                }
            }
        }

        __syncthreads();

        if ((threadIdx.x == 0) && (condition == 1))
            counter_smem = atomicAdd(counter, counter_smem);
        __syncthreads();

        if ((selec_smem[tid3] > 0) && (condition ==1)){
            sub_spData_dat[counter_smem + threadIdx.x] = selec_smem[tid3];
            sub_spData_row[counter_smem + threadIdx.x] = selec_smem[tid3 + 1];
            sub_spData_col[counter_smem + threadIdx.x] = selec_smem[tid3 + 2];
        }
    }




    __global__ void flip_frag(frag* fragArray,frag* o_fragArray, int id_f_flip,
                              int n_frags, float2 subfrags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag < n_frags){

            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            // UDPATE
            int sub_pos_fi = o_fragArray->sub_pos[id_frag];
            // UDPATE
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            // UDPATE
            int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];
            // UDPATE
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            // UDPATE
            int sub_len_fi = o_fragArray->sub_len[id_frag];
            // UDPATE
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];

            fragArray->pos[id_frag] = pos_fi;

            fragArray->sub_pos[id_frag] = sub_pos_fi;

            fragArray->id_c[id_frag] = contig_fi;
            fragArray->start_bp[id_frag] = start_bp_fi;
            fragArray->len_bp[id_frag] = len_bp_fi;
            // UDPATE
            fragArray->sub_len[id_frag] = sub_len_fi;
            // UDPATE
            fragArray->circ[id_frag] = circ_fi;
            fragArray->id[id_frag] = id_frag;
            if (id_frag == id_f_flip){
                fragArray->ori[id_frag] = or_fi * -1;
            }
            else{
                fragArray->ori[id_frag] = or_fi;
            }
            fragArray->prev[id_frag] = id_prev_fi;
            fragArray->next[id_frag] = id_next_fi;
            fragArray->l_cont[id_frag] = l_cont_fi;
            // UDPATE
            fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;
            // UDPATE
            fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
            fragArray->rep[id_frag] = rep_fi;
            fragArray->activ[id_frag] = activ_fi;
            fragArray->id_d[id_frag] = id_d_fi;
        }
    }



    __global__ void swap_activity_frag(frag* fragArray,frag* o_fragArray, int id_f_unactiv, int max_id_contig,
                              int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag < n_frags){

            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            // UDPATE
            int sub_pos_fi = o_fragArray->sub_pos[id_frag];
            // UDPATE
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            // UDPATE
            int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];
            // UDPATE
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            // UDPATE
            int sub_len_fi = o_fragArray->sub_len[id_frag];
            // UDPATE
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];

            fragArray->pos[id_frag] = pos_fi;

            fragArray->sub_pos[id_frag] = sub_pos_fi;

            fragArray->start_bp[id_frag] = start_bp_fi;
            fragArray->len_bp[id_frag] = len_bp_fi;
            fragArray->circ[id_frag] = circ_fi;
            fragArray->id[id_frag] = id_frag;
            fragArray->ori[id_frag] = or_fi;
//            if ((id_frag == id_f_unactiv) && (id_d_fi != id_frag)){
            if ((id_frag == id_f_unactiv) && (rep_fi == 1)){
                fragArray->activ[id_frag] = 0 * (activ_fi == 1) + 1 * (activ_fi == 0);
                fragArray->id_c[id_frag] = contig_fi * (activ_fi == 1) + (max_id_contig + 1) * (activ_fi == 0);
//                fragArray->id_c[id_frag] = contig_fi * (activ_fi == 1) + (max_id_contig + 1) * (activ_fi == 0);
            }
            else{
                fragArray->activ[id_frag] = activ_fi;
                fragArray->id_c[id_frag] = contig_fi;
            }
            fragArray->prev[id_frag] = id_prev_fi;
            fragArray->next[id_frag] = id_next_fi;
            fragArray->l_cont[id_frag] = l_cont_fi;
            // UDPATE
            fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;
            // UDPATE
            fragArray->sub_len[id_frag] = sub_len_fi;
            // UDPATE
            fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
            fragArray->rep[id_frag] = rep_fi;
            fragArray->id_d[id_frag] = id_d_fi;
        }
    }


    __global__ void pop_out_frag(frag* fragArray,frag* o_fragArray, int* pop_id_contigs, int id_f_pop,
                                 int max_id_contig, int n_frags)
    {
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        // UDPATE
        __shared__ int sub_pos_f_pop;
        // UDPATE
        __shared__ int l_cont_f_pop;
            // UDPATE
        __shared__ int sub_l_cont_f_pop;
            // UDPATE
        __shared__ int l_cont_bp_f_pop;
        __shared__ int len_bp_f_pop;
        // UDPATE
        __shared__ int sub_len_f_pop;
        // UDPATE
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int or_f_pop;
        __shared__ int circ_f_pop;


        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            // UDPATE
            sub_pos_f_pop = o_fragArray->sub_pos[id_f_pop];
            // UDPATE
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            // UDPATE
            sub_l_cont_f_pop = o_fragArray->sub_l_cont[id_f_pop];
            // UDPATE
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            // UDPATE
            sub_len_f_pop = o_fragArray->sub_len[id_f_pop];
            // UDPATE
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            or_f_pop = o_fragArray->ori[id_f_pop];
            circ_f_pop = o_fragArray->circ[id_f_pop];
        }
        __syncthreads();

        if (id_frag < n_frags){
            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            // UDPATE
            int sub_pos_fi = o_fragArray->sub_pos[id_frag];
            // UDPATE
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            // UDPATE
            int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];
            // UDPATE
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            // UDPATE
            int sub_len_fi = o_fragArray->sub_len[id_frag];
            // UDPATE
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            if (l_cont_f_pop > 2){
                if ( contig_fi == contig_f_pop){
                    if (pos_fi < pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi;
                        // UDPATE
                        fragArray->sub_pos[id_frag] = sub_pos_fi;
                        // UDPATE
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;

                        fragArray->sub_len[id_frag] = sub_len_fi;

                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
//                        fragArray->prev[id_frag] = id_prev_fi;
                        if ((id_frag == id_next_f_pop) && (circ_f_pop == 1)){
                            fragArray->prev[id_frag] = id_prev_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        if (pos_fi == (pos_f_pop - 1)){
                            fragArray->next[id_frag] = id_next_f_pop;
                        }
                        else{
                            fragArray->next[id_frag] = id_next_fi;
                        }
                        fragArray->l_cont[id_frag] = l_cont_fi -1;

                        fragArray->sub_l_cont[id_frag] = sub_l_cont_fi - sub_len_f_pop;

                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                    else if (pos_fi == pos_f_pop){
                        fragArray->pos[id_frag] = 0;
                        // UDPATE
                        fragArray->sub_pos[id_frag] = 0;
                        // UDPATE
                        fragArray->id_c[id_frag] = max_id_contig + 1;
                        pop_id_contigs[id_frag] = max_id_contig + 1;
                        fragArray->start_bp[id_frag] = 0;
                        fragArray->len_bp[id_frag] = len_bp_fi;

                        fragArray->sub_len[id_frag] = sub_len_fi;

                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = 1;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = 1;

                        fragArray->sub_l_cont[id_frag] = sub_len_fi;

                        fragArray->l_cont_bp[id_frag] = len_bp_fi;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi > pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi - 1;
                        // UDPATE
                        fragArray->sub_pos[id_frag] = sub_pos_fi - sub_len_f_pop;
                        // UDPATE
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi - len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        // UDPATE
                        fragArray->sub_len[id_frag] = sub_len_fi;
                        // UDPATE
                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if (pos_fi == (pos_f_pop + 1)){
                            fragArray->prev[id_frag] = id_prev_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        if ((id_frag == id_prev_f_pop) && (circ_f_pop == 1)){
                            fragArray->next[id_frag] = id_next_f_pop;
                        }
                        else{
                            fragArray->next[id_frag] = id_next_fi;
                        }
//                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_fi -1 ;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_fi - sub_len_f_pop;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }

                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    // UDPATE
                    fragArray->sub_pos[id_frag] = sub_pos_fi;
                    // UDPATE
                    fragArray->id_c[id_frag] = contig_fi;
                    pop_id_contigs[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    // UDPATE
                    fragArray->sub_len[id_frag] = sub_len_fi;
                    // UDPATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    // UDPATE
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;
                    // UDPATE
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;

                }
            }
            else if (l_cont_f_pop == 2){
                if ( contig_fi == contig_f_pop){
                    if (pos_fi < pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi;
                        // UDPATE
                        fragArray->sub_pos[id_frag] = sub_pos_fi;
                        // UDPATE
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        // UDPATE
                        fragArray->sub_len[id_frag] = sub_len_fi;
                        // UDPATE
                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = l_cont_fi -1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        // UDPATE
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_fi - sub_len_f_pop;
                        // UDPATE
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                    else if (pos_fi == pos_f_pop){
                        fragArray->pos[id_frag] = 0;
                        // UDPATE
                        fragArray->sub_pos[id_frag] = 0;
                        // UDPATE
                        fragArray->id_c[id_frag] = max_id_contig + 1;
                        pop_id_contigs[id_frag] = max_id_contig + 1;
                        fragArray->start_bp[id_frag] = 0;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        // UDPATE
                        fragArray->sub_len[id_frag] = sub_len_fi;
                        // UDPATE
                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = 1;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = 1;
                        fragArray->l_cont_bp[id_frag] = len_bp_fi;
                        // UDPATE
                        fragArray->sub_l_cont[id_frag] = sub_len_fi;
                        // UDPATE
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi > pos_f_pop){
                        fragArray->pos[id_frag] = pos_fi - 1;
                        // UDPATE
                        fragArray->sub_pos[id_frag] = sub_pos_fi - sub_len_f_pop;
                        // UDPATE
                        fragArray->id_c[id_frag] = contig_fi;
                        pop_id_contigs[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi - len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        // UDPATE
                        fragArray->sub_len[id_frag] = sub_len_fi;
                        // UDPATE
                        fragArray->circ[id_frag] = 0;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = -1;
                        fragArray->next[id_frag] = -1;
                        fragArray->l_cont[id_frag] = l_cont_fi -1 ;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi - len_bp_f_pop;
                        // UDPATE
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_fi - sub_len_f_pop;
                        // UDPATE
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }

                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    // UDPATE
                    fragArray->sub_pos[id_frag] = sub_pos_fi;
                    // UDPATE
                    fragArray->id_c[id_frag] = contig_fi;
                    pop_id_contigs[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    // UDPATE
                    fragArray->sub_len[id_frag] = sub_len_fi;
                    // UDPATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    // UDPATE
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;
                    // UDPATE
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;

                }
            }
            else{
                fragArray->pos[id_frag] = pos_fi;
                // UDPATE
                fragArray->sub_pos[id_frag] = sub_pos_fi;
                // UDPATE
                fragArray->id_c[id_frag] = contig_fi;
                pop_id_contigs[id_frag] = contig_fi;
                fragArray->start_bp[id_frag] = start_bp_fi;
                fragArray->len_bp[id_frag] = len_bp_fi;
                // UDPATE
                fragArray->sub_len[id_frag] = sub_len_fi;
                // UDPATE
                fragArray->circ[id_frag] = circ_fi;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = or_fi;
                fragArray->prev[id_frag] = id_prev_fi;
                fragArray->next[id_frag] = id_next_fi;
                fragArray->l_cont[id_frag] = l_cont_fi;
                // UDPATE
                fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;
                // UDPATE
                fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                fragArray->rep[id_frag] = rep_fi;
                fragArray->activ[id_frag] = activ_fi;
                fragArray->id_d[id_frag] = id_d_fi;
            }
        }
    }


    __global__ void pop_in_frag_1(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
                                  int ori_f_pop,
                                  int n_frags)
    // split insert @ left
    {
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int sub_pos_f_pop;             // UDPATE
        __shared__ int l_cont_f_pop;
        __shared__ int sub_l_cont_f_pop;            // UDPATE
        __shared__ int l_cont_bp_f_pop;
        __shared__ int len_bp_f_pop;
        __shared__ int sub_len_f_pop;             // UDPATE
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int activ_f_pop;
//        __shared__ int or_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int sub_pos_f_ins;              // UDPATE
        __shared__ int l_cont_f_ins;
        __shared__ int sub_l_cont_f_ins;              // UDPATE
        __shared__ int l_cont_bp_f_ins;
        __shared__ int len_bp_f_ins;
        __shared__ int sub_len_f_ins;              // UDPATE
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            sub_pos_f_pop = o_fragArray->sub_pos[id_f_pop];              // UDPATE
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            sub_l_cont_f_pop = o_fragArray->sub_l_cont[id_f_pop];              // UDPATE
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            sub_len_f_pop = o_fragArray->sub_len[id_f_pop];              // UDPATE
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            sub_pos_f_ins = o_fragArray->sub_pos[id_f_ins];              // UDPATE
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            sub_l_cont_f_ins = o_fragArray->sub_l_cont[id_f_ins];              // UDPATE
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            sub_len_f_ins = o_fragArray->sub_len[id_f_ins];              // UDPATE
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
        }
        __syncthreads();


        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
            if (id_frag == id_f_pop){
                fragArray->pos[id_frag] = 0;
                fragArray->sub_pos[id_frag] = 0;              // UDPATE
                fragArray->start_bp[id_frag] = 0;
                fragArray->len_bp[id_frag] = len_bp_f_pop;
                fragArray->sub_len[id_frag] = sub_len_f_pop;              // UDPATE
                fragArray->circ[id_frag] = 0;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = ori_f_pop;
                fragArray->prev[id_frag] = -1;
                fragArray->next[id_frag] = id_f_ins;
                if (circ_f_ins == 0){
                    fragArray->id_c[id_frag] = max_id_contig + 1;
                    fragArray->l_cont[id_frag] = l_cont_f_ins - pos_f_ins + 1;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + len_bp_f_pop;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins - sub_pos_f_ins + sub_len_f_pop;              // UDPATE
                }
                else{
                    fragArray->id_c[id_frag] = contig_f_ins;
                    fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;
                }
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int sub_pos_fi = o_fragArray->sub_pos[id_frag];              // UDPATE
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];              // UDPATE
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int sub_len_fi = o_fragArray->sub_len[id_frag];              // UDPATE
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];

                if (contig_fi == contig_f_ins){
                    if (circ_f_ins == 0){
                        if (pos_fi < pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (pos_fi == (pos_f_ins -1)){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = pos_f_ins;
                            fragArray->sub_l_cont[id_frag] = sub_pos_f_ins;  // UPDATE
                            fragArray->l_cont_bp[id_frag] = start_bp_f_ins;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = 1;
                            fragArray->sub_pos[id_frag] = sub_len_f_pop;  // UPDATE
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_f_ins;
                            fragArray->prev[id_frag] = id_f_pop;
                            fragArray->next[id_frag] = id_next_f_ins;
                            fragArray->l_cont[id_frag] = l_cont_f_ins - pos_f_ins + 1;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins - sub_pos_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + len_bp_f_pop;
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - (pos_f_ins) + 1;
                            fragArray->sub_pos[id_frag] = sub_pos_fi - (sub_pos_f_ins) + sub_len_f_pop;  // UPDATE
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_ins) + len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins - pos_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins - sub_pos_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                    }
                    else{ // contig_f_ins is circular
                        if (pos_fi < pos_f_ins){
                                fragArray->pos[id_frag] = l_cont_f_ins - pos_f_ins + pos_fi + 1;
                                fragArray->sub_pos[id_frag] = sub_l_cont_f_ins - sub_pos_f_ins + sub_pos_fi + sub_len_f_pop;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_ins;
                                fragArray->start_bp[id_frag] = l_cont_bp_f_ins - start_bp_f_ins + start_bp_fi + len_bp_f_pop;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (pos_fi == pos_f_ins - 1){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
                                fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                                fragArray->rep[id_frag] = rep_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = 1;
                            fragArray->sub_pos[id_frag] = sub_len_f_pop; // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_f_ins;
                            fragArray->sub_len[id_frag] = sub_len_f_ins; // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_f_ins;
                            fragArray->prev[id_frag] = id_f_pop;
                            fragArray->next[id_frag] = id_next_f_ins;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - pos_f_ins + 1;
                            fragArray->sub_pos[id_frag] = sub_pos_fi - sub_pos_f_ins + sub_len_f_pop; // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_ins + len_bp_f_pop;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (id_frag == id_prev_f_ins){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
//                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;  // UPDATE
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;
                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
                fragArray->sub_pos[id_frag] = o_fragArray->sub_pos[id_frag];  // UPDATE
                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
                fragArray->sub_len[id_frag] = o_fragArray->sub_len[id_frag];  // UPDATE
                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
                fragArray->next[id_frag] = o_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
                fragArray->sub_l_cont[id_frag] = o_fragArray->sub_l_cont[id_frag];  // UPDATE
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
        }
    }

    __global__ void pop_in_frag_2(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
                                  int ori_f_pop,
                                  int n_frags)
    {
    // split insert @ right
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int sub_pos_f_pop;  // UPDATE
        __shared__ int l_cont_f_pop;
        __shared__ int l_cont_bp_f_pop;
        __shared__ int sub_l_cont_f_pop;  // UPDATE
        __shared__ int len_bp_f_pop;
        __shared__ int sub_len_f_pop;  // UPDATE
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int activ_f_pop;
//        __shared__ int or_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int sub_pos_f_ins; // UPDATE
        __shared__ int l_cont_f_ins;
        __shared__ int l_cont_bp_f_ins;
        __shared__ int sub_l_cont_f_ins;  // UPDATE
        __shared__ int len_bp_f_ins;
        __shared__ int sub_len_f_ins;  // UPDATE
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            sub_pos_f_pop = o_fragArray->sub_pos[id_f_pop];  // UPDATE
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            sub_l_cont_f_pop = o_fragArray->sub_l_cont[id_f_pop];  // UPDATE
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            sub_len_f_pop = o_fragArray->sub_len[id_f_pop];  // UPDATE
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            sub_pos_f_ins = o_fragArray->sub_pos[id_f_ins];  // UPDATE
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            sub_l_cont_f_ins = o_fragArray->sub_l_cont[id_f_ins];  // UPDATE
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            sub_len_f_ins = o_fragArray->sub_len[id_f_ins];  // UPDATE
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
        }
        __syncthreads();
        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
            if (id_frag == id_f_pop){
                if (circ_f_ins == 0){
                    fragArray->pos[id_frag] = pos_f_ins + 1;
                    fragArray->sub_pos[id_frag] = sub_pos_f_ins + sub_len_f_ins;  // UPDATE
                    fragArray->id_c[id_frag] = contig_f_ins;
                    fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_ins;
                    fragArray->len_bp[id_frag] = len_bp_f_pop;
                    fragArray->sub_len[id_frag] = sub_len_f_pop;  // UPDATE
                    fragArray->circ[id_frag] = 0;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = ori_f_pop;
                    fragArray->prev[id_frag] = id_f_ins;
                    fragArray->next[id_frag] = -1;
                    fragArray->l_cont[id_frag] = pos_f_ins + 2;
                    fragArray->l_cont_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + len_bp_f_pop;
                    fragArray->sub_l_cont[id_frag] = sub_pos_f_ins + sub_len_f_ins + sub_len_f_pop;  // UPDATE
                    fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                    fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                    fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
                }
                else{
                    fragArray->pos[id_frag] = (l_cont_f_ins - (pos_f_ins  + 1)) + pos_f_ins + 1;
                    fragArray->sub_pos[id_frag] = (sub_l_cont_f_ins - (sub_pos_f_ins  + sub_len_f_ins))
                                                    + sub_pos_f_ins + sub_len_f_ins;  // UPDATE
                    fragArray->id_c[id_frag] = contig_f_ins;
                    fragArray->start_bp[id_frag] = (l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins))
                                                    + start_bp_f_ins + len_bp_f_ins;
                    fragArray->len_bp[id_frag] = len_bp_f_pop;
                    fragArray->sub_len[id_frag] = sub_len_f_pop;  // UPDATE
                    fragArray->circ[id_frag] = 0;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = ori_f_pop;
                    fragArray->prev[id_frag] = id_f_ins;
                    fragArray->next[id_frag] = -1;
                    fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                    fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                    fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                    fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
                }
            }
            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int sub_pos_fi = o_fragArray->sub_pos[id_frag];  // UPDATE
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag]; // UPDATE
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int sub_len_fi = o_fragArray->sub_len[id_frag]; // UPDATE
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];
                if (contig_fi == contig_f_ins){
                    if(circ_f_ins == 0){
                        if (pos_fi < pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = pos_f_ins + 2;
                            fragArray->l_cont_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_pos_f_ins + sub_len_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;
                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_f_ins;
                            fragArray->prev[id_frag] = id_prev_f_ins;
                            fragArray->next[id_frag] = id_f_pop;
                            fragArray->l_cont[id_frag] = pos_f_ins + 2;
                            fragArray->l_cont_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_pos_f_ins + sub_len_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - (pos_f_ins + 1);
                            fragArray->sub_pos[id_frag] = sub_pos_fi - (sub_pos_f_ins + sub_len_f_ins);  // UPDATE
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_ins + len_bp_f_ins);
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == (pos_f_ins + 1)){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins - (pos_f_ins + 1);
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins);
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins - (sub_pos_f_ins + sub_len_f_ins);  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                    }
                    else{//circular contig
                        if (pos_fi < pos_f_ins){
                            fragArray->pos[id_frag] = (l_cont_f_ins - (pos_f_ins  + 1)) + pos_fi;
                            fragArray->sub_pos[id_frag] = (sub_l_cont_f_ins - (sub_pos_f_ins  + sub_len_f_ins))
                                                            + sub_pos_fi;  // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = (l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins))
                                                            + start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (id_frag == id_next_f_ins){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
//                            fragArray->prev[id_frag] = id_prev_fi;
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                        else if (pos_fi == pos_f_ins){
                            fragArray->pos[id_frag] = (l_cont_f_ins - (pos_f_ins  + 1)) + pos_f_ins;
                            fragArray->sub_pos[id_frag] = (sub_l_cont_f_ins - (sub_pos_f_ins  + sub_len_f_ins))
                                                            + sub_pos_f_ins;  // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = (l_cont_bp_f_ins - (start_bp_f_ins + len_bp_f_ins))
                                                            + start_bp_f_ins;
                            fragArray->len_bp[id_frag] = len_bp_f_ins;
                            fragArray->sub_len[id_frag] = sub_len_f_ins;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_f_ins;
                            fragArray->next[id_frag] = id_f_pop;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                        else if (pos_fi > pos_f_ins){
                            fragArray->pos[id_frag] = pos_fi - (pos_f_ins + 1);
                            fragArray->sub_pos[id_frag] = sub_pos_fi - (sub_pos_f_ins + sub_len_f_ins);  // UPDATE
                            fragArray->id_c[id_frag] = contig_f_ins;
                            fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_ins + len_bp_f_ins);
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_f_ins +1){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                            fragArray->rep[id_frag] = rep_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->id_d[id_frag] = id_d_fi;

                        }
                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;  // UPDATE
                    fragArray->rep[id_frag] = rep_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->id_d[id_frag] = id_d_fi;

                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
                fragArray->sub_pos[id_frag] = o_fragArray->sub_pos[id_frag];  // UPDATE
                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
                fragArray->sub_len[id_frag] = o_fragArray->sub_len[id_frag];  // UPDATE
                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
                fragArray->next[id_frag] = o_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
                fragArray->sub_l_cont[id_frag] = o_fragArray->sub_l_cont[id_frag];  // UPDATE
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
        }
    }

    __global__ void pop_in_frag_3(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
                                  int ori_f_pop,
                                  int n_frags)
    // insert frag @ right of id_f_ins
    {
        __shared__ int contig_f_pop;
        __shared__ int pos_f_pop;
        __shared__ int sub_pos_f_pop;  // UPDATE
        __shared__ int l_cont_f_pop;
        __shared__ int l_cont_bp_f_pop;
        __shared__ int sub_l_cont_f_pop;  // UPDATE
        __shared__ int len_bp_f_pop;
        __shared__ int sub_len_f_pop;  // UPDATE
        __shared__ int start_bp_f_pop;
        __shared__ int id_prev_f_pop;
        __shared__ int id_next_f_pop;
        __shared__ int activ_f_pop;
//        __shared__ int or_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int sub_pos_f_ins;  // UPDATE
        __shared__ int l_cont_f_ins;
        __shared__ int l_cont_bp_f_ins;
        __shared__ int sub_l_cont_f_ins;  // UPDATE
        __shared__ int len_bp_f_ins;
        __shared__ int sub_len_f_ins;  // UPDATE
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            pos_f_pop = o_fragArray->pos[id_f_pop];
            sub_pos_f_pop = o_fragArray->sub_pos[id_f_pop];  // UPDATE
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            sub_l_cont_f_pop = o_fragArray->sub_l_cont[id_f_pop];  // UPDATE
            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
            sub_len_f_pop = o_fragArray->sub_len[id_f_pop];  // UPDATE
            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
            id_prev_f_pop = o_fragArray->prev[id_f_pop];
            id_next_f_pop = o_fragArray->next[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            sub_pos_f_ins = o_fragArray->sub_pos[id_f_ins];  // UPDATE
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            sub_l_cont_f_ins = o_fragArray->sub_l_cont[id_f_ins];  // UPDATE
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            sub_len_f_ins = o_fragArray->sub_len[id_f_ins];  // UPDATE
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
        }
        __syncthreads();
        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
            if (id_frag == id_f_pop){
                fragArray->pos[id_frag] = pos_f_ins + 1;
                fragArray->sub_pos[id_frag] = sub_pos_f_ins + sub_len_f_ins;  // UPDATE
                fragArray->id_c[id_frag] = contig_f_ins;
                fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_ins;
                fragArray->len_bp[id_frag] = len_bp_f_pop;
                fragArray->sub_len[id_frag] = sub_len_f_pop;  // UPDATE
                fragArray->circ[id_frag] = circ_f_ins;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = ori_f_pop;
                fragArray->prev[id_frag] = id_f_ins;
                fragArray->next[id_frag] = id_next_f_ins;
                fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int sub_pos_fi = o_fragArray->sub_pos[id_frag];  // UPDATE
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];  // UPDATE
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int sub_len_fi = o_fragArray->sub_len[id_frag];  // UPDATE
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];
                if (contig_fi == contig_f_ins){
                    if (pos_fi < pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if ((id_frag == id_next_f_ins) && ( circ_f_ins == 1)){
                            fragArray->prev[id_frag] = id_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop; // UPDATE
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi == pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_f_ins;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_f_pop;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                    else if (pos_fi > pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi + 1;
                        fragArray->sub_pos[id_frag] = sub_pos_fi + sub_len_f_pop;  // UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi + len_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if (pos_fi == (pos_f_ins + 1)){
                            fragArray->prev[id_frag] = id_f_pop;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;

                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;  // UPDATE
                    fragArray->id_d[id_frag] = id_d_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->rep[id_frag] = rep_fi;

                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
                fragArray->sub_pos[id_frag] = o_fragArray->sub_pos[id_frag];  // UPDATE
                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
                fragArray->sub_len[id_frag] = o_fragArray->sub_len[id_frag];  // UPDATE
                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
                fragArray->next[id_frag] = o_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
                fragArray->sub_l_cont[id_frag] = o_fragArray->sub_l_cont[id_frag];  // UPDATE
                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
            }
        }
    }

//    __global__ void pop_in_frag_4(frag* fragArray,frag* o_fragArray, int id_f_pop, int id_f_ins, int max_id_contig,
//                                  int ori_f_pop,
//                                  int n_frags)
//    // insert frag @ left of id_f_ins
//    {
//        __shared__ int contig_f_pop;
//        __shared__ int pos_f_pop;
//        __shared__ int sub_pos_f_pop;  // UPDATE
//        __shared__ int l_cont_f_pop;
//        __shared__ int l_cont_bp_f_pop;
//        __shared__ int sub_l_cont_f_pop;  // UPDATE
//        __shared__ int len_bp_f_pop;
//        __shared__ int sub_len_f_pop;  // UPDATE
//        __shared__ int start_bp_f_pop;
//        __shared__ int id_prev_f_pop;
//        __shared__ int id_next_f_pop;
//        __shared__ int activ_f_pop;
////        __shared__ int or_f_pop;
//
//        __shared__ int contig_f_ins;
//        __shared__ int pos_f_ins;
//        __shared__ int sub_pos_f_ins;  // UPDATE
//        __shared__ int l_cont_f_ins;
//        __shared__ int l_cont_bp_f_ins;
//        __shared__ int sub_l_cont_f_ins;  // UPDATE
//        __shared__ int len_bp_f_ins;
//        __shared__ int sub_len_f_ins;  // UPDATE
//        __shared__ int start_bp_f_ins;
//        __shared__ int id_prev_f_ins;
//        __shared__ int id_next_f_ins;
//        __shared__ int circ_f_ins;
//        __shared__ int or_f_ins;
//        __shared__ int activ_f_ins;
//
//        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
//        if (threadIdx.x == 0){
//            contig_f_pop = o_fragArray->id_c[id_f_pop];
//            pos_f_pop = o_fragArray->pos[id_f_pop];
//            sub_pos_f_pop = o_fragArray->sub_pos[id_f_pop];  // UPDATE
//            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
//            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
//            sub_l_cont_f_pop = o_fragArray->sub_l_cont[id_f_pop];  // UPDATE
//            len_bp_f_pop = o_fragArray->len_bp[id_f_pop];
//            sub_len_f_pop = o_fragArray->sub_len[id_f_pop];  // UPDATE
//            start_bp_f_pop = o_fragArray->start_bp[id_f_pop];
//            id_prev_f_pop = o_fragArray->prev[id_f_pop];
//            id_next_f_pop = o_fragArray->next[id_f_pop];
//            activ_f_pop = o_fragArray->activ[id_f_pop];
//
//            contig_f_ins = o_fragArray->id_c[id_f_ins];
//            pos_f_ins = o_fragArray->pos[id_f_ins];
//            sub_pos_f_ins = o_fragArray->sub_pos[id_f_ins];  // UPDATE
//            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
//            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
//            sub_l_cont_f_ins = o_fragArray->sub_l_cont[id_f_ins];  // UPDATE
//            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
//            sub_len_f_ins = o_fragArray->sub_len[id_f_ins];  // UPDATE
//            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
//            id_prev_f_ins = o_fragArray->prev[id_f_ins];
//            id_next_f_ins = o_fragArray->next[id_f_ins];
//            circ_f_ins = o_fragArray->circ[id_f_ins];
//            or_f_ins = o_fragArray->ori[id_f_ins];
//            activ_f_ins = o_fragArray->activ[id_f_ins];
//        }
//        __syncthreads();
//        if ((activ_f_ins == 1) && ( activ_f_pop == 1)){
//            if (id_frag == id_f_pop){
//                fragArray->pos[id_frag] = pos_f_ins ;
//                fragArray->sub_pos[id_frag] = sub_pos_f_ins ;  // UPDATE
//                fragArray->id_c[id_frag] = contig_f_ins;
//                fragArray->start_bp[id_frag] = start_bp_f_ins;
//                fragArray->len_bp[id_frag] = len_bp_f_pop;
//                fragArray->sub_len[id_frag] = sub_len_f_pop;  // UPDATE
//                fragArray->circ[id_frag] = circ_f_ins;
//                fragArray->id[id_frag] = id_frag;
//                fragArray->ori[id_frag] = ori_f_pop;
//                fragArray->prev[id_frag] = id_prev_f_ins;
//                fragArray->next[id_frag] = id_f_ins;
//                fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
//                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
//                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
//                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
//                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
//                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
//
//            }
//            else if ((id_frag < n_frags) && (id_frag != id_f_pop)){
//                int contig_fi = o_fragArray->id_c[id_frag];
//                int pos_fi = o_fragArray->pos[id_frag];
//                int sub_pos_fi = o_fragArray->sub_pos[id_frag];  // UPDATE
//                int l_cont_fi = o_fragArray->l_cont[id_frag];
//                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
//                int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];  // UPDATE
//                int len_bp_fi = o_fragArray->len_bp[id_frag];
//                int sub_len_fi = o_fragArray->sub_len[id_frag];  // UPDATE
//                int circ_fi = o_fragArray->circ[id_frag];
//                int id_prev_fi = o_fragArray->prev[id_frag];
//                int id_next_fi = o_fragArray->next[id_frag];
//                int start_bp_fi = o_fragArray->start_bp[id_frag];
//                int or_fi = o_fragArray->ori[id_frag];
//                int rep_fi = o_fragArray->rep[id_frag];
//                int activ_fi = o_fragArray->activ[id_frag];
//                int id_d_fi = o_fragArray->id_d[id_frag];
//
//                if (contig_fi == contig_f_ins){
//                    if (pos_fi < pos_f_ins){
//                        fragArray->pos[id_frag] = pos_fi;
//                        fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
//                        fragArray->id_c[id_frag] = contig_f_ins;
//                        fragArray->start_bp[id_frag] = start_bp_fi;
//                        fragArray->len_bp[id_frag] = len_bp_fi;
//                        fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
//                        fragArray->circ[id_frag] = circ_f_ins;
//                        fragArray->id[id_frag] = id_frag;
//                        fragArray->ori[id_frag] = or_fi;
//                        fragArray->prev[id_frag] = id_prev_fi;
//                        if (pos_fi == pos_f_ins -1){
//                            fragArray->next[id_frag] = id_f_pop;
//                        }
//                        else{
//                            fragArray->next[id_frag] = id_next_fi;
//                        }
//                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
//                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
//                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
//                        fragArray->id_d[id_frag] = id_d_fi;
//                        fragArray->activ[id_frag] = activ_fi;
//                        fragArray->rep[id_frag] = rep_fi;
//
//                    }
//                    else if (pos_fi == pos_f_ins){
//                        fragArray->pos[id_frag] = pos_f_ins + 1;
//                        fragArray->sub_pos[id_frag] = sub_pos_f_ins + sub_len_f_pop;  // UPDATE
//                        fragArray->id_c[id_frag] = contig_f_ins;
//                        fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_pop;
//                        fragArray->len_bp[id_frag] = len_bp_fi;
//                        fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
//                        fragArray->circ[id_frag] = circ_f_ins;
//                        fragArray->id[id_frag] = id_frag;
//                        fragArray->ori[id_frag] = or_f_ins;
//                        fragArray->prev[id_frag] = id_f_pop;
//                        fragArray->next[id_frag] = id_next_f_ins;
//                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
//                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
//                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
//                        fragArray->id_d[id_frag] = id_d_fi;
//                        fragArray->activ[id_frag] = activ_fi;
//                        fragArray->rep[id_frag] = rep_fi;
//
//                    }
//                    else if (pos_fi > pos_f_ins){
//                        fragArray->pos[id_frag] = pos_fi + 1;
//                        fragArray->sub_pos[id_frag] = sub_pos_fi + sub_len_f_pop;  // UPDATE
//                        fragArray->id_c[id_frag] = contig_f_ins;
//                        fragArray->start_bp[id_frag] = start_bp_fi + len_bp_f_pop;
//                        fragArray->len_bp[id_frag] = len_bp_fi;
//                        fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
//                        fragArray->circ[id_frag] = circ_f_ins;
//                        fragArray->id[id_frag] = id_frag;
//                        fragArray->ori[id_frag] = or_fi;
//                        fragArray->prev[id_frag] = id_prev_fi;
//                        fragArray->next[id_frag] = id_next_fi;
//                        fragArray->l_cont[id_frag] = l_cont_f_ins + 1;
//                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + len_bp_f_pop;
//                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_len_f_pop;  // UPDATE
//                        fragArray->id_d[id_frag] = id_d_fi;
//                        fragArray->activ[id_frag] = activ_fi;
//                        fragArray->rep[id_frag] = rep_fi;
//
//                    }
//                }
//                else{
//                    fragArray->pos[id_frag] = pos_fi;
//                    fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
//                    fragArray->id_c[id_frag] = contig_fi;
//                    fragArray->start_bp[id_frag] = start_bp_fi;
//                    fragArray->len_bp[id_frag] = len_bp_fi;
//                    fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
//                    fragArray->circ[id_frag] = circ_fi;
//                    fragArray->id[id_frag] = id_frag;
//                    fragArray->ori[id_frag] = or_fi;
//                    fragArray->prev[id_frag] = id_prev_fi;
//                    fragArray->next[id_frag] = id_next_fi;
//                    fragArray->l_cont[id_frag] = l_cont_fi;
//                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
//                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;  // UPDATE
//                    fragArray->id_d[id_frag] = id_d_fi;
//                    fragArray->activ[id_frag] = activ_fi;
//                    fragArray->rep[id_frag] = rep_fi;
//
//                }
//            }
//        }
//        else{
//            if (id_frag < n_frags){
//                fragArray->pos[id_frag] = o_fragArray->pos[id_frag];
//                fragArray->sub_pos[id_frag] = o_fragArray->sub_pos[id_frag];  // UPDATE
//                fragArray->id_c[id_frag] = o_fragArray->id_c[id_frag];
//                fragArray->circ[id_frag] = o_fragArray->circ[id_frag];
//                fragArray->id[id_frag] = id_frag;
//                fragArray->ori[id_frag] = o_fragArray->ori[id_frag];
//                fragArray->start_bp[id_frag] = o_fragArray->start_bp[id_frag];
//                fragArray->len_bp[id_frag] = o_fragArray->len_bp[id_frag];
//                fragArray->sub_len[id_frag] = o_fragArray->sub_len[id_frag];  // UPDATE
//                fragArray->prev[id_frag] = o_fragArray->prev[id_frag];
//                fragArray->next[id_frag] = o_fragArray->next[id_frag];
//                fragArray->l_cont[id_frag] = o_fragArray->l_cont[id_frag];
//                fragArray->l_cont_bp[id_frag] = o_fragArray->l_cont_bp[id_frag];
//                fragArray->sub_l_cont[id_frag] = o_fragArray->sub_l_cont[id_frag];  // UPDATE
//                fragArray->rep[id_frag] = o_fragArray->rep[id_frag];
//                fragArray->activ[id_frag] = o_fragArray->activ[id_frag];
//                fragArray->id_d[id_frag] = o_fragArray->id_d[id_frag];
//            }
//        }
//    }


    __global__ void get_bounds(frag* fragArray,
                               int id_f_pop,
                               int id_f_ins,
                               int* list_valid_insert,
                               int* list_bounds,
                               int* id_f_cut_upstream,
                               int* id_f_cut_downstream,
                               int n_bounds,
                               int n_frags)
    // get id frag for block extraction and fill list of valid mutations and uniq mutations
    {
        __shared__ int contig_f_pop;
        __shared__ int contig_f_ins;
        __shared__ int list_pos_cut_up[N_TO_CUT];
        __shared__ int list_pos_cut_down[N_TO_CUT];
        int pos_cut_up;
        int pos_cut_down;
        int i, size_extract = 0;
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        int thread_can_write = id_frag == 0;
        int same_contigs, ins_is_ext, pop_is_ext, l_cont_f_pop, l_cont_f_ins, pos_f_pop, pos_f_ins;
        if (threadIdx.x == 0){
            contig_f_pop = fragArray->id_c[id_f_pop];
            contig_f_ins = fragArray->id_c[id_f_ins];

            same_contigs = (contig_f_pop == contig_f_ins);

            pos_f_pop = fragArray->pos[id_f_pop];
            pos_f_ins = fragArray->pos[id_f_ins];

            l_cont_f_pop = fragArray->l_cont[id_f_pop];
            l_cont_f_ins = fragArray->l_cont[id_f_ins];

            ins_is_ext = (pos_f_ins == 0) || (pos_f_ins == (l_cont_f_ins - 1));
            pop_is_ext = (pos_f_pop == 0) || (pos_f_pop == (l_cont_f_pop - 1));

            for(i = 0; i < n_bounds; i++){
                 if (i == 0){
                    if (same_contigs){ // local flip
                        if ((pos_f_ins < pos_f_pop - 1)) { // cut upstream and intra flip ok
                            pos_cut_up = pos_f_ins + 1;
                            pos_cut_down = pos_f_pop;
                        }
                        else if (pos_f_ins > pos_f_pop + 1){ // cut downstream and intra flip ok
                            pos_cut_down = pos_f_ins - 1;
                            pos_cut_up = pos_f_pop;
                        }
                        else{
                            pos_cut_up = pos_f_pop;
                            pos_cut_down = pos_f_pop;
                        }
                    }
                    else{
                        pos_cut_up = pos_f_pop;
                        pos_cut_down = pos_f_pop;
                    }
                 }
                 else if ((i > 0) && ( i < n_bounds - 1)){ // inter intra insert block
                    pos_cut_up = max(0 , pos_f_pop - list_bounds[i - 1]);
                    pos_cut_down = min(l_cont_f_pop - 1, pos_f_pop + list_bounds[i - 1]);
                 }
                 else{ // inter intra big insert
                    pos_cut_up = 0;
                    pos_cut_down = l_cont_f_pop - 1;
                 }
                 ///////////////////////////////////////////////////////////////////////////////////////////////////////
                 if (same_contigs && (pos_f_ins <= pos_f_pop) && (pos_f_ins >=pos_cut_up)){ // test validity
                     list_pos_cut_up[i] = -1;
                     if (thread_can_write){
                         list_valid_insert[i * 2] = -1;
                     }
                 }
                 else{ // test on unicity of the mutation
                     list_pos_cut_up[i] = pos_cut_up;
                     if (pos_cut_up == 0){
                        size_extract = pos_f_pop - pos_cut_up;
                        if ((size_extract == 1) || (ins_is_ext)){
                            if (thread_can_write){
                                list_valid_insert[i * 2] = -1;
                            }
                            list_pos_cut_up[i] = -1;

                        }
                        else{
                            if (thread_can_write){
                                list_valid_insert[i * 2] = 1;
                            }
                        }
                     }
                     else{
                        if (thread_can_write){
                            list_valid_insert[i * 2] = 1;
                        }
                     }

                 }
                 ///////////////////////////////////////////////////////////////////////////////////////////////////////
                 if (same_contigs && (((pos_f_ins >= pos_f_pop) && (pos_f_ins <= pos_cut_down)) || (pos_f_ins == (pos_f_pop - 1)))){// test validity
                     list_pos_cut_down[i] = -1;
                     if (thread_can_write){
                         list_valid_insert[i * 2 + 1] = -1;
                     }
                 }
                 else{ // test on unicity of the mutation
                     list_pos_cut_down[i] = pos_cut_down;
                     if (pos_cut_down == l_cont_f_pop - 1){
                        size_extract = pos_cut_down - pos_f_pop;
                        if ((size_extract == 1) || (ins_is_ext)){
                            if (thread_can_write){
                                list_valid_insert[i * 2 + 1] = -1;
                            }
                            list_pos_cut_down[i] = -1;
                        }
                        else{
                            if (thread_can_write){
                                list_valid_insert[i * 2 + 1] = 1;
                            }
                        }
                     }
                     else{
                        if (thread_can_write){
                            list_valid_insert[i * 2 + 1] = 1;
                        }
                     }
                 }

            }
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }
        __syncthreads();

        if (id_frag < n_frags){
            if (fragArray->id_c[id_frag] == contig_f_pop){
                for (i = 0; i < n_bounds; i++){
//                    id_f_cut_upstream[i] = -1;
//                    id_f_cut_downstream[i] = -1;

                    if (fragArray->pos[id_frag] == list_pos_cut_down[i]){
                        id_f_cut_downstream[i] = id_frag;
                    }
                    if (fragArray->pos[id_frag] == list_pos_cut_up[i]){
                        id_f_cut_upstream[i] = id_frag;
                    }
                }
            }
        }
    }

//    __global__ void get_bounds(frag* fragArray,
//                               int id_f_pop,
//                               int id_f_ins,
//                               int* list_valid_insert,
//                               int* list_bounds,
//                               int* id_f_cut_upstream,
//                               int* id_f_cut_downstream,
//                               int n_bounds,
//                               int n_frags)
//    // get id frag for block extraction and fill list of valid mutations and uniq mutations
//    {
//        __shared__ int contig_f_pop;
//        __shared__ int contig_f_ins;
//        __shared__ int list_pos_cut_up[N_TO_CUT];
//        __shared__ int list_pos_cut_down[N_TO_CUT];
//
//        int pos_cut_up;
//        int pos_cut_down;
//        int i, size_extract = 0;
//        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
//        int thread_can_write = id_frag == 0;
//        int same_contigs, ins_is_ext, pop_is_ext, l_cont_f_pop, l_cont_f_ins, pos_f_pop, pos_f_ins;
//        if (threadIdx.x == 0){
//            contig_f_pop = fragArray->id_c[id_f_pop];
//            contig_f_ins = fragArray->id_c[id_f_ins];
//
//            same_contigs = (contig_f_pop == contig_f_ins);
//
//            pos_f_pop = fragArray->pos[id_f_pop];
//            pos_f_ins = fragArray->pos[id_f_ins];
//            l_cont_f_pop = fragArray->l_cont[id_f_pop];
//            l_cont_f_ins = fragArray->l_cont[id_f_ins];
//
//            ins_is_ext = (pos_f_ins == 0) || (pos_f_ins == (l_cont_f_ins - 1));
//            pop_is_ext = (pos_f_pop == 0) || (pos_f_pop == (l_cont_f_pop - 1));
//
//            for(i = 0; i < n_bounds; i++){
//                 if (i != n_bounds -1){
//                     pos_cut_up = max(0 , pos_f_pop - list_bounds[i]);
//                     pos_cut_down = min(l_cont_f_pop - 1, pos_f_pop + list_bounds[i]);
//                 }
//                 else{
//                    pos_cut_up = 0;
//                    pos_cut_down = l_cont_f_pop - 1;
//                 }
//                 ///////////////////////////////////////////////////////////////////////////////////////////////////////
//                 if (same_contigs && (pos_f_ins <= pos_f_pop) && (pos_f_ins >=pos_cut_up)){ // test validity
//                     list_pos_cut_up[i] = -1;
//                     if (thread_can_write){
//                         list_valid_insert[i * 2] = -1;
//                     }
//                 }
//                 else{ // test on unicity of the mutation
//                     list_pos_cut_up[i] = pos_cut_up;
//                     if (pos_cut_up == 0){
//                        size_extract = pos_f_pop - pos_cut_up;
//                        if ((size_extract == 1) || (ins_is_ext)){
//                            if (thread_can_write){
//                                list_valid_insert[i * 2] = -1;
//                            }
//                            list_pos_cut_up[i] = -1;
//
//                        }
//                        else{
//                            if (thread_can_write){
//                                list_valid_insert[i * 2] = 1;
//                            }
//                        }
//                     }
//                     else{
//                        if (thread_can_write){
//                            list_valid_insert[i * 2] = 1;
//                        }
//                     }
//
//                 }
//                 ///////////////////////////////////////////////////////////////////////////////////////////////////////
//                 if (same_contigs && (((pos_f_ins >= pos_f_pop) && (pos_f_ins <= pos_cut_down)) || (pos_f_ins == (pos_f_pop - 1)))){
//                     list_pos_cut_down[i] = -1;
//                     if (thread_can_write){
//                         list_valid_insert[i * 2 + 1] = -1;
//                     }
//                 }
//                 else{ // test on unicity of the mutation
//                     list_pos_cut_down[i] = pos_cut_down;
//                     if (pos_cut_down == l_cont_f_pop - 1){
//                        size_extract = pos_cut_down - pos_f_pop;
//                        if ((size_extract == 1) || (ins_is_ext)){
//                            if (thread_can_write){
//                                list_valid_insert[i * 2 + 1] = -1;
//                            }
//                            list_pos_cut_down[i] = -1;
//                        }
//                        else{
//                            if (thread_can_write){
//                                list_valid_insert[i * 2 + 1] = 1;
//                            }
//                        }
//                     }
//                     else{
//                        if (thread_can_write){
//                            list_valid_insert[i * 2 + 1] = 1;
//                        }
//                     }
//                 }
//
//            }
//            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        }
//        __syncthreads();
//
//        if (id_frag < n_frags){
//            if (fragArray->id_c[id_frag] == contig_f_pop){
//                for (i = 0; i < n_bounds; i++){
////                    id_f_cut_upstream[i] = -1;
////                    id_f_cut_downstream[i] = -1;
//
//                    if (fragArray->pos[id_frag] == list_pos_cut_down[i]){
//                        id_f_cut_downstream[i] = id_frag;
//                    }
//                    if (fragArray->pos[id_frag] == list_pos_cut_up[i]){
//                        id_f_cut_upstream[i] = id_frag;
//                    }
//                }
//            }
//        }
//    }

    __global__ void extract_block(frag* fragArray,
                                  frag* o_fragArray,
                                  int* split_id_contigs,
                                  int id_f_cut_a,
                                  int* list_id_f_cut_b,
                                  int id_fb,
                                  int upstream,
                                  int max_id_contig,
                                  int n_frags)
    {
        __shared__ int contig_f_cut;
        __shared__ int l_cont_f_cut;
        __shared__ int sub_l_cont_f_cut; //UPDATE
        __shared__ int l_cont_bp_f_cut;

        __shared__ int circ_f_cut;

        __shared__ int pos_f_cut_a;
        __shared__ int sub_pos_f_cut_a; //UPDATE
        __shared__ int len_bp_f_cut_a;
        __shared__ int sub_len_f_cut_a; //UPDATE
        __shared__ int start_bp_f_cut_a;
        __shared__ int id_prev_f_cut_a;
        __shared__ int id_next_f_cut_a;
        __shared__ int or_f_cut_a;
        __shared__ int activ_f_cut_a;

        __shared__ int pos_f_cut_b;
        __shared__ int sub_pos_f_cut_b; //UPDATE
        __shared__ int len_bp_f_cut_b;
        __shared__ int sub_len_f_cut_b; //UPDATE
        __shared__ int start_bp_f_cut_b;
        __shared__ int id_prev_f_cut_b;
        __shared__ int id_next_f_cut_b;
        __shared__ int or_f_cut_b;
        __shared__ int activ_f_cut_b;

        __shared__ int id_f_cut_b;
        __shared__ int size_extract;
        __shared__ int sub_size_extract;
        __shared__ int size_extract_bp;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;

        if (threadIdx.x == 0){


            contig_f_cut = o_fragArray->id_c[id_f_cut_a];
            l_cont_f_cut = o_fragArray->l_cont[id_f_cut_a];
            sub_l_cont_f_cut = o_fragArray->sub_l_cont[id_f_cut_a]; // UPDATE
            l_cont_bp_f_cut = o_fragArray->l_cont_bp[id_f_cut_a];
            circ_f_cut = o_fragArray->circ[id_f_cut_a];

            pos_f_cut_a = o_fragArray->pos[id_f_cut_a];
            sub_pos_f_cut_a = o_fragArray->sub_pos[id_f_cut_a]; //UPDATE
            len_bp_f_cut_a = o_fragArray->len_bp[id_f_cut_a];
            sub_len_f_cut_a = o_fragArray->sub_len[id_f_cut_a]; //UPDATE
            start_bp_f_cut_a = o_fragArray->start_bp[id_f_cut_a];
            id_prev_f_cut_a = o_fragArray->prev[id_f_cut_a];
            id_next_f_cut_a = o_fragArray->next[id_f_cut_a];
            or_f_cut_a = o_fragArray->ori[id_f_cut_a];
            activ_f_cut_a = o_fragArray->activ[id_f_cut_a];

            id_f_cut_b = list_id_f_cut_b[id_fb];
            if (id_f_cut_b >= 0){
                pos_f_cut_b = o_fragArray->pos[id_f_cut_b];
                sub_pos_f_cut_b = o_fragArray->sub_pos[id_f_cut_b]; //UPDATE
                len_bp_f_cut_b = o_fragArray->len_bp[id_f_cut_b];
                sub_len_f_cut_b = o_fragArray->sub_len[id_f_cut_b]; //UPDATE
                start_bp_f_cut_b = o_fragArray->start_bp[id_f_cut_b];
                id_prev_f_cut_b = o_fragArray->prev[id_f_cut_b];
                id_next_f_cut_b = o_fragArray->next[id_f_cut_b];
                or_f_cut_b = o_fragArray->ori[id_f_cut_b];
                activ_f_cut_b = o_fragArray->activ[id_f_cut_b];

                if (upstream == 1){
                    size_extract = pos_f_cut_a - pos_f_cut_b + 1;
                    sub_size_extract = sub_pos_f_cut_a - sub_pos_f_cut_b + sub_len_f_cut_a;

                    size_extract_bp = start_bp_f_cut_a - start_bp_f_cut_b + len_bp_f_cut_a;
                }
                else{
                    size_extract = pos_f_cut_b - pos_f_cut_a + 1;
                    sub_size_extract = sub_pos_f_cut_b - sub_pos_f_cut_a + sub_len_f_cut_b;
                    size_extract_bp = start_bp_f_cut_b - start_bp_f_cut_a + len_bp_f_cut_b;
                }
            }
            else{
                activ_f_cut_b = 0;
            }
        }
        __syncthreads();
        if (id_frag < n_frags){
            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            int sub_pos_fi = o_fragArray->sub_pos[id_frag]; //UPDATE
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag]; //UPDATE
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            int sub_len_fi = o_fragArray->sub_len[id_frag]; //UPDATE
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];
//            if ((activ_f_cut_a == 1) && (activ_f_cut_b == 1) && (l_cont_f_cut > 2)){
            if ((activ_f_cut_a == 1) && (activ_f_cut_b == 1)){
                if (contig_fi == contig_f_cut){
                    if (upstream == 1){
                        if (pos_fi < pos_f_cut_b){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                            fragArray->id_c[id_frag] = contig_f_cut;
                            split_id_contigs[id_frag] = contig_f_cut;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                            fragArray->circ[id_frag] = circ_f_cut;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (pos_fi == pos_f_cut_b - 1){
                                fragArray->next[id_frag] = id_next_f_cut_a;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_f_cut - size_extract;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut - sub_size_extract; //UPDATE
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - size_extract_bp;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                        else if((pos_fi >= pos_f_cut_b) && (pos_fi <= pos_f_cut_a)){
                            fragArray->pos[id_frag] = pos_fi - pos_f_cut_b;
                            fragArray->sub_pos[id_frag] = sub_pos_fi - sub_pos_f_cut_b; //UPDATE
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            split_id_contigs[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_cut_b;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_f_cut_b){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            if (pos_fi == pos_f_cut_a){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = size_extract;
                            fragArray->sub_l_cont[id_frag] = sub_size_extract; //UPDATE
                            fragArray->l_cont_bp[id_frag] = size_extract_bp;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                        else if (pos_fi > pos_f_cut_a){
                            fragArray->pos[id_frag] = pos_fi - size_extract;
                            fragArray->sub_pos[id_frag] = sub_pos_fi - sub_size_extract; //UPDATE
                            fragArray->id_c[id_frag] = contig_f_cut;
                            split_id_contigs[id_frag] = contig_f_cut;
                            fragArray->start_bp[id_frag] = start_bp_fi - size_extract_bp;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; //UDPATE
                            fragArray->circ[id_frag] = circ_f_cut;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_f_cut_a + 1){
                                fragArray->prev[id_frag] = id_prev_f_cut_b;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }

                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_cut  - size_extract;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut  - sub_size_extract; //UPDATE
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - size_extract_bp;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                    }
                    else {
                        if (pos_fi < pos_f_cut_a){
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi; //UPDATE
                            fragArray->id_c[id_frag] = contig_f_cut;
                            split_id_contigs[id_frag] = contig_f_cut;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                            fragArray->circ[id_frag] = circ_f_cut;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (pos_fi == pos_f_cut_a - 1){
                                fragArray->next[id_frag] = id_next_f_cut_b;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_f_cut - size_extract;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut - sub_size_extract; //UPDATE
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - size_extract_bp;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                        else if((pos_fi >= pos_f_cut_a) && (pos_fi <= pos_f_cut_b)){
                            fragArray->pos[id_frag] = pos_fi - pos_f_cut_a;
                            fragArray->sub_pos[id_frag] = sub_pos_fi - sub_pos_f_cut_a; //UPDATE
                            fragArray->id_c[id_frag] = max_id_contig + 1;
                            split_id_contigs[id_frag] = max_id_contig + 1;
                            fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_cut_a;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_f_cut_a){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            if (pos_fi == pos_f_cut_b){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = size_extract;
                            fragArray->sub_l_cont[id_frag] = sub_size_extract; //UPDATE
                            fragArray->l_cont_bp[id_frag] = size_extract_bp;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                        else if (pos_fi > pos_f_cut_b){
                            fragArray->pos[id_frag] = pos_fi - size_extract;
                            fragArray->sub_pos[id_frag] = sub_pos_fi - sub_size_extract;//UPDATE
                            fragArray->id_c[id_frag] = contig_f_cut;
                            split_id_contigs[id_frag] = contig_f_cut;
                            fragArray->start_bp[id_frag] = start_bp_fi - size_extract_bp;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                            fragArray->circ[id_frag] = circ_f_cut;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_f_cut_b + 1){
                                fragArray->prev[id_frag] = id_prev_f_cut_a;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_f_cut  - size_extract;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut  - sub_size_extract; //UPDATE
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - size_extract_bp;
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                    }

                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->sub_pos[id_frag] = sub_pos_fi; //UPDATE
                    fragArray->id_c[id_frag] = contig_fi;
                    split_id_contigs[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi; //UPDATE
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->id_d[id_frag] = id_d_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->rep[id_frag] = rep_fi;
                }
            }
            else{
                fragArray->pos[id_frag] = pos_fi;
                fragArray->sub_pos[id_frag] = sub_pos_fi; //UPDATE
                fragArray->id_c[id_frag] = contig_fi;
                split_id_contigs[id_frag] = contig_fi;
                fragArray->start_bp[id_frag] = start_bp_fi;
                fragArray->len_bp[id_frag] = len_bp_fi;
                fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                fragArray->circ[id_frag] = circ_fi;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = or_fi;
                fragArray->prev[id_frag] = id_prev_fi;
                fragArray->next[id_frag] = id_next_fi;
                fragArray->l_cont[id_frag] = l_cont_fi;
                fragArray->sub_l_cont[id_frag] = sub_l_cont_fi; //UPDATE
                fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                fragArray->id_d[id_frag] = id_d_fi;
                fragArray->activ[id_frag] = activ_fi;
                fragArray->rep[id_frag] = rep_fi;
            }
        }
    }


    __global__ void insert_block(frag* fragArray,
                                 frag* o_fragArray,
                                 frag* init_fragArray,
                                 int id_f_pop,
                                 int id_f_ins,
                                 int* list_id_bounds,
                                 int* list_valid_insert,
                                 int id_mutation,
                                 int id_bound,
                                 int upstream,
                                 int n_frags)
    // insert contig which frag belong  @ right of id_f_ins !!!!
    {
        __shared__ int contig_f_pop;
        __shared__ int l_cont_f_pop;
        __shared__ int sub_l_cont_f_pop;//UPDATE
        __shared__ int l_cont_bp_f_pop;
        __shared__ int activ_f_pop;

        __shared__ int contig_f_ins;
        __shared__ int pos_f_ins;
        __shared__ int sub_pos_f_ins;//UPDATE
        __shared__ int l_cont_f_ins;
        __shared__ int sub_l_cont_f_ins; //UPDATE
        __shared__ int l_cont_bp_f_ins;
        __shared__ int len_bp_f_ins;
        __shared__ int sub_len_f_ins; //UPDATE
        __shared__ int start_bp_f_ins;
        __shared__ int id_prev_f_ins;
        __shared__ int id_next_f_ins;
        __shared__ int circ_f_ins;
        __shared__ int or_f_ins;
        __shared__ int activ_f_ins;
        __shared__ int id_extremity;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_pop = o_fragArray->id_c[id_f_pop];
            l_cont_f_pop = o_fragArray->l_cont[id_f_pop];
            sub_l_cont_f_pop = o_fragArray->sub_l_cont[id_f_pop];//UPDATE
            l_cont_bp_f_pop = o_fragArray->l_cont_bp[id_f_pop];
            activ_f_pop = o_fragArray->activ[id_f_pop];

            contig_f_ins = o_fragArray->id_c[id_f_ins];
            pos_f_ins = o_fragArray->pos[id_f_ins];
            sub_pos_f_ins = o_fragArray->sub_pos[id_f_ins]; //UPDATE
            l_cont_f_ins = o_fragArray->l_cont[id_f_ins];
            sub_l_cont_f_ins = o_fragArray->sub_l_cont[id_f_ins]; //UPDATE
            l_cont_bp_f_ins = o_fragArray->l_cont_bp[id_f_ins];
            len_bp_f_ins = o_fragArray->len_bp[id_f_ins];
            sub_len_f_ins = o_fragArray->sub_len[id_f_ins]; //UPDATE
            start_bp_f_ins = o_fragArray->start_bp[id_f_ins];
            id_prev_f_ins = o_fragArray->prev[id_f_ins];
            id_next_f_ins = o_fragArray->next[id_f_ins];
            circ_f_ins = o_fragArray->circ[id_f_ins];
            or_f_ins = o_fragArray->ori[id_f_ins];
            activ_f_ins = o_fragArray->activ[id_f_ins];
            id_extremity = list_id_bounds[id_bound];
        }
        __syncthreads();
        if ((activ_f_ins == 1) && ( activ_f_pop == 1) && (contig_f_ins != contig_f_pop) && (list_valid_insert[id_mutation] != -1)){
            if (id_frag < n_frags){
                int contig_fi = o_fragArray->id_c[id_frag];
                int pos_fi = o_fragArray->pos[id_frag];
                int sub_pos_fi = o_fragArray->sub_pos[id_frag]; //UPDATE
                int l_cont_fi = o_fragArray->l_cont[id_frag];
                int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];//UPDATE
                int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
                int len_bp_fi = o_fragArray->len_bp[id_frag];
                int sub_len_fi = o_fragArray->sub_len[id_frag]; //UPDATE
                int circ_fi = o_fragArray->circ[id_frag];
                int id_prev_fi = o_fragArray->prev[id_frag];
                int id_next_fi = o_fragArray->next[id_frag];
                int start_bp_fi = o_fragArray->start_bp[id_frag];
                int or_fi = o_fragArray->ori[id_frag];
                int rep_fi = o_fragArray->rep[id_frag];
                int activ_fi = o_fragArray->activ[id_frag];
                int id_d_fi = o_fragArray->id_d[id_frag];
                if (contig_fi == contig_f_ins){
                    if (pos_fi < pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->sub_pos[id_frag] = sub_pos_fi; //UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if ((id_frag == id_next_f_ins) && ( circ_f_ins == 1)){
                            fragArray->prev[id_frag] = id_extremity;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + l_cont_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_l_cont_f_pop; //UPDATE
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + l_cont_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                    else if (pos_fi == pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->sub_pos[id_frag] = sub_pos_fi; //UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_f_ins;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_f_pop;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + l_cont_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_l_cont_f_pop; //UPDATE
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + l_cont_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                    else if (pos_fi > pos_f_ins){
                        fragArray->pos[id_frag] = pos_fi + l_cont_f_pop;
                        fragArray->sub_pos[id_frag] = sub_pos_fi + sub_l_cont_f_pop; //UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_fi + l_cont_bp_f_pop;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if (pos_fi == (pos_f_ins + 1)){
                            fragArray->prev[id_frag] = id_extremity;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_f_ins + l_cont_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_l_cont_f_pop; //UPDATE
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + l_cont_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                }
                else if (contig_fi == contig_f_pop){
                    if (upstream == 0){
                        fragArray->pos[id_frag] = pos_f_ins + 1 + pos_fi;
                        fragArray->sub_pos[id_frag] = sub_pos_f_ins + sub_len_f_ins + sub_pos_fi; //UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        if (pos_fi == 0){
                            fragArray->prev[id_frag] = id_f_ins;
                        }
                        else{
                            fragArray->prev[id_frag] = id_prev_fi;
                        }
                        if (pos_fi == l_cont_fi - 1){
                            fragArray->next[id_frag] = id_next_f_ins;
                        }
                        else{
                            fragArray->next[id_frag] = id_next_fi;
                        }
                        fragArray->l_cont[id_frag] = l_cont_f_ins + l_cont_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_l_cont_f_pop; //UPDATE
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + l_cont_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                    else{
                        fragArray->pos[id_frag] = pos_f_ins + 1 + (l_cont_f_pop - pos_fi - 1);
                        fragArray->sub_pos[id_frag] = sub_pos_f_ins + sub_len_f_ins + (sub_l_cont_f_pop - sub_pos_fi -
                                                                                       sub_len_fi); //UPDATE
                        fragArray->id_c[id_frag] = contig_f_ins;
                        fragArray->start_bp[id_frag] = start_bp_f_ins + len_bp_f_ins + (l_cont_bp_f_pop - start_bp_fi -
                                                                                        len_bp_fi);
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                        fragArray->circ[id_frag] = circ_f_ins;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi * -1;
                        if (pos_fi == l_cont_fi - 1){
                            fragArray->prev[id_frag] = id_f_ins;
                        }
                        else{
                            fragArray->prev[id_frag] = id_next_fi;
                        }
                        if (pos_fi == 0){
                            fragArray->next[id_frag] = id_next_f_ins;
                        }
                        else{
                            fragArray->next[id_frag] = id_prev_fi;
                        }
                        fragArray->l_cont[id_frag] = l_cont_f_ins + l_cont_f_pop;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_f_ins + sub_l_cont_f_pop; //UPDATE
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_f_ins + l_cont_bp_f_pop;
                        fragArray->rep[id_frag] = rep_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->id_d[id_frag] = id_d_fi;
                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->sub_pos[id_frag] = sub_pos_fi; //UPDATE
                    fragArray->id_c[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->sub_len[id_frag] = sub_len_fi; //UPDATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi; //UPDATE
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->id_d[id_frag] = id_d_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->rep[id_frag] = rep_fi;

                }
            }
        }
        else{
            if (id_frag < n_frags){
                fragArray->pos[id_frag] = init_fragArray->pos[id_frag];
                fragArray->sub_pos[id_frag] = init_fragArray->sub_pos[id_frag]; //UPDATE
                fragArray->id_c[id_frag] = init_fragArray->id_c[id_frag];
                fragArray->circ[id_frag] = init_fragArray->circ[id_frag];
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = init_fragArray->ori[id_frag];
                fragArray->start_bp[id_frag] = init_fragArray->start_bp[id_frag];
                fragArray->len_bp[id_frag] = init_fragArray->len_bp[id_frag];
                fragArray->sub_len[id_frag] = init_fragArray->sub_len[id_frag];  //UPDATE
                fragArray->prev[id_frag] = init_fragArray->prev[id_frag];
                fragArray->next[id_frag] = init_fragArray->next[id_frag];
                fragArray->l_cont[id_frag] = init_fragArray->l_cont[id_frag];
                fragArray->sub_l_cont[id_frag] = init_fragArray->sub_l_cont[id_frag];  //UPDATE
                fragArray->l_cont_bp[id_frag] = init_fragArray->l_cont_bp[id_frag];
                fragArray->rep[id_frag] = init_fragArray->rep[id_frag];
                fragArray->activ[id_frag] = init_fragArray->activ[id_frag];
                fragArray->id_d[id_frag] = init_fragArray->id_d[id_frag];
            }
        }
    }


    __global__ void split_contig(frag* fragArray,frag* o_fragArray, int* split_id_contigs, int id_f_cut, int upstream,
                                 int max_id_contig, int n_frags)
    {
        __shared__ int contig_f_cut;
        __shared__ int pos_f_cut;
        __shared__ int sub_pos_f_cut;  // UPDATE
        __shared__ int l_cont_f_cut;
        __shared__ int l_cont_bp_f_cut;
        __shared__ int sub_l_cont_f_cut;  // UPDATE
        __shared__ int len_bp_f_cut;
        __shared__ int sub_len_f_cut;  // UPDATE
        __shared__ int start_bp_f_cut;
        __shared__ int id_prev_f_cut;
        __shared__ int id_next_f_cut;
        __shared__ int circ_f_cut;
        __shared__ int or_f_cut;
        __shared__ int activ_f_cut;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_f_cut = o_fragArray->id_c[id_f_cut];
            pos_f_cut = o_fragArray->pos[id_f_cut];
            sub_pos_f_cut = o_fragArray->sub_pos[id_f_cut];  // UPDATE
            l_cont_f_cut = o_fragArray->l_cont[id_f_cut];
            l_cont_bp_f_cut = o_fragArray->l_cont_bp[id_f_cut];
            sub_l_cont_f_cut = o_fragArray->sub_l_cont[id_f_cut];  // UPDATE
            len_bp_f_cut = o_fragArray->len_bp[id_f_cut];
            sub_len_f_cut = o_fragArray->sub_len[id_f_cut];  // UPDATE
            start_bp_f_cut = o_fragArray->start_bp[id_f_cut];
            id_prev_f_cut = o_fragArray->prev[id_f_cut];
            id_next_f_cut = o_fragArray->next[id_f_cut];
            circ_f_cut = o_fragArray->circ[id_f_cut];
            or_f_cut = o_fragArray->ori[id_f_cut];
            activ_f_cut = o_fragArray->activ[id_f_cut];
        }
        __syncthreads();
        if (id_frag < n_frags){
            int contig_fi = o_fragArray->id_c[id_frag];
            int pos_fi = o_fragArray->pos[id_frag];
            int sub_pos_fi = o_fragArray->sub_pos[id_frag]; // UPDATE
            int l_cont_fi = o_fragArray->l_cont[id_frag];
            int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
            int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag]; // UPDATE
            int len_bp_fi = o_fragArray->len_bp[id_frag];
            int sub_len_fi = o_fragArray->sub_len[id_frag]; // UPDATE
            int circ_fi = o_fragArray->circ[id_frag];
            int id_prev_fi = o_fragArray->prev[id_frag];
            int id_next_fi = o_fragArray->next[id_frag];
            int start_bp_fi = o_fragArray->start_bp[id_frag];
            int or_fi = o_fragArray->ori[id_frag];
            int rep_fi = o_fragArray->rep[id_frag];
            int activ_fi = o_fragArray->activ[id_frag];
            int id_d_fi = o_fragArray->id_d[id_frag];
            if ((activ_f_cut == 1) && (l_cont_f_cut > 1)){
                if (contig_fi == contig_f_cut){
                    if (circ_f_cut == 0){ // linear contig
                        if (upstream == 1){
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi;
                                fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (pos_fi == pos_f_cut - 1){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
                                fragArray->l_cont[id_frag] = pos_f_cut;
                                fragArray->l_cont_bp[id_frag] = start_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_pos_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if(pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = 0;
                                fragArray->sub_pos[id_frag] = 0;  // UPDATE
                                fragArray->id_c[id_frag] = max_id_contig + 1;
                                split_id_contigs[id_frag] = max_id_contig + 1;
                                fragArray->start_bp[id_frag] = 0;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->sub_len[id_frag] = sub_len_f_cut; // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = -1;
                                fragArray->next[id_frag] = id_next_f_cut;
                                fragArray->l_cont[id_frag] = l_cont_f_cut  - pos_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - start_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut - sub_pos_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - pos_f_cut;
                                fragArray->sub_pos[id_frag] = sub_pos_fi - sub_pos_f_cut;  // UPDATE
                                fragArray->id_c[id_frag] = max_id_contig + 1;
                                split_id_contigs[id_frag] = max_id_contig + 1;
                                fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut  - pos_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - start_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut - sub_pos_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                        else{
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi;
                                fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = pos_f_cut + 1;
                                fragArray->l_cont_bp[id_frag] = start_bp_f_cut + len_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_pos_f_cut + sub_len_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if(pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = pos_f_cut;
                                fragArray->sub_pos[id_frag] = sub_pos_f_cut;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->sub_len[id_frag] = sub_len_f_cut;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_f_cut;
                                fragArray->next[id_frag] = -1;
                                fragArray->l_cont[id_frag] = pos_f_cut + 1;
                                fragArray->l_cont_bp[id_frag] = start_bp_f_cut + len_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_pos_f_cut + sub_len_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - (pos_f_cut + 1);
                                fragArray->sub_pos[id_frag] = sub_pos_fi - (sub_pos_f_cut + sub_len_f_cut);  // UPDATE
                                fragArray->id_c[id_frag] = max_id_contig + 1;
                                split_id_contigs[id_frag] = max_id_contig +1;
                                fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_cut + len_bp_f_cut);
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                if (pos_fi == pos_f_cut + 1){
                                    fragArray->prev[id_frag] = -1;
                                }
                                else{
                                    fragArray->prev[id_frag] = id_prev_fi;
                                }
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut - (pos_f_cut + 1);
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut - (start_bp_f_cut + len_bp_f_cut);
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut - (sub_pos_f_cut + sub_len_f_cut);  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                    }
                    else{ // circular contig
                        if (upstream ==1){
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = l_cont_f_cut - pos_f_cut + pos_fi;
                                fragArray->sub_pos[id_frag] = sub_l_cont_f_cut - sub_pos_f_cut + sub_pos_fi;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = l_cont_bp_f_cut - start_bp_f_cut + start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (pos_fi == pos_f_cut - 1){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = 0;
                                fragArray->sub_pos[id_frag] = 0;
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = 0;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->sub_len[id_frag] = sub_len_f_cut;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = -1;
                                fragArray->next[id_frag] = id_next_f_cut;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut;
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - pos_f_cut;
                                fragArray->sub_pos[id_frag] = sub_pos_fi - sub_pos_f_cut;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi - start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_fi;  // UPDATE
                                fragArray->sub_len[id_frag] = sub_len_fi;
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_fi;
                                if (id_frag == id_prev_f_cut){
                                    fragArray->next[id_frag] = -1;
                                }
                                else{
                                    fragArray->next[id_frag] = id_next_fi;
                                }
//                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                        else{
                            if (pos_fi < pos_f_cut){
                                fragArray->pos[id_frag] = (l_cont_f_cut - (pos_f_cut  + 1)) + pos_fi;
                                fragArray->sub_pos[id_frag] = (sub_l_cont_f_cut - (sub_pos_f_cut  + sub_len_f_cut))
                                                                + sub_pos_fi;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = (l_cont_bp_f_cut - (start_bp_f_cut + len_bp_f_cut))
                                                                + start_bp_fi;
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
//                                fragArray->prev[id_frag] = id_prev_fi;
                                if (id_frag == id_next_f_cut){
                                    fragArray->prev[id_frag] = -1;
                                }
                                else{
                                    fragArray->prev[id_frag] = id_prev_fi;
                                }
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi == pos_f_cut){
                                fragArray->pos[id_frag] = (l_cont_f_cut - (pos_f_cut  + 1)) + pos_fi;
                                fragArray->sub_pos[id_frag] = (sub_l_cont_f_cut - (sub_pos_f_cut  + sub_len_f_cut))
                                                                + sub_pos_f_cut;  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = (l_cont_bp_f_cut - (start_bp_f_cut + len_bp_f_cut))
                                                                + start_bp_f_cut;
                                fragArray->len_bp[id_frag] = len_bp_f_cut;
                                fragArray->sub_len[id_frag] = sub_len_f_cut;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                fragArray->prev[id_frag] = id_prev_f_cut;
                                fragArray->next[id_frag] = -1;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                            else if (pos_fi > pos_f_cut){
                                fragArray->pos[id_frag] = pos_fi - (pos_f_cut + 1);
                                fragArray->sub_pos[id_frag] = sub_pos_fi - (sub_pos_f_cut + sub_len_f_cut);  // UPDATE
                                fragArray->id_c[id_frag] = contig_f_cut;
                                split_id_contigs[id_frag] = contig_f_cut;
                                fragArray->start_bp[id_frag] = start_bp_fi - (start_bp_f_cut + len_bp_f_cut);
                                fragArray->len_bp[id_frag] = len_bp_fi;
                                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                                fragArray->circ[id_frag] = 0;
                                fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi;
                                if (pos_fi == pos_f_cut +1){
                                    fragArray->prev[id_frag] = -1;
                                }
                                else{
                                    fragArray->prev[id_frag] = id_prev_fi;
                                }
                                fragArray->next[id_frag] = id_next_fi;
                                fragArray->l_cont[id_frag] = l_cont_f_cut;
                                fragArray->l_cont_bp[id_frag] = l_cont_bp_f_cut;
                                fragArray->sub_l_cont[id_frag] = sub_l_cont_f_cut;  // UPDATE
                                fragArray->id_d[id_frag] = id_d_fi;
                                fragArray->activ[id_frag] = activ_fi;
                                fragArray->rep[id_frag] = rep_fi;
                            }
                        }
                    }
                }
                else{
                    fragArray->pos[id_frag] = pos_fi;
                    fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                    fragArray->id_c[id_frag] = contig_fi;
                    split_id_contigs[id_frag] = contig_fi;
                    fragArray->start_bp[id_frag] = start_bp_fi;
                    fragArray->len_bp[id_frag] = len_bp_fi;
                    fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                    fragArray->circ[id_frag] = circ_fi;
                    fragArray->id[id_frag] = id_frag;
                    fragArray->ori[id_frag] = or_fi;
                    fragArray->prev[id_frag] = id_prev_fi;
                    fragArray->next[id_frag] = id_next_fi;
                    fragArray->l_cont[id_frag] = l_cont_fi;
                    fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                    fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;  // UPDATE
                    fragArray->id_d[id_frag] = id_d_fi;
                    fragArray->activ[id_frag] = activ_fi;
                    fragArray->rep[id_frag] = rep_fi;
                }
            }
            else{
                fragArray->pos[id_frag] = pos_fi;
                fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                fragArray->id_c[id_frag] = contig_fi;
                split_id_contigs[id_frag] = contig_fi;
                fragArray->start_bp[id_frag] = start_bp_fi;
                fragArray->len_bp[id_frag] = len_bp_fi;
                fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                fragArray->circ[id_frag] = circ_fi;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = or_fi;
                fragArray->prev[id_frag] = id_prev_fi;
                fragArray->next[id_frag] = id_next_fi;
                fragArray->l_cont[id_frag] = l_cont_fi;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                fragArray->sub_l_cont[id_frag] = sub_l_cont_fi;  // UPDATE
                fragArray->id_d[id_frag] = id_d_fi;
                fragArray->activ[id_frag] = activ_fi;
                fragArray->rep[id_frag] = rep_fi;
            }
        }
    }

    __global__ void paste_contigs(frag* fragArray,frag* o_fragArray, int id_fA, int id_fB, int max_id_contig,
                                  int n_frags)
    {
        __shared__ int contig_fA;
        __shared__ int pos_fA;
        __shared__ int sub_pos_fA;  // UPDATE
        __shared__ int l_cont_fA;
        __shared__ int l_cont_bp_fA;
        __shared__ int sub_l_cont_fA;  // UPDATE
        __shared__ int len_bp_fA;
        __shared__ int sub_len_fA;  // UPDATE
        __shared__ int start_bp_fA;
        __shared__ int id_prev_fA;
        __shared__ int id_next_fA;
        __shared__ int circ_fA;
        __shared__ int or_fA;
        __shared__ int activ_fA;

        __shared__ int contig_fB;
        __shared__ int pos_fB;
        __shared__ int sub_pos_fB;  // UPDATE
        __shared__ int l_cont_fB;
        __shared__ int l_cont_bp_fB;
        __shared__ int sub_l_cont_fB;  // UPDATE
        __shared__ int len_bp_fB;
        __shared__ int sub_len_fB;  // UPDATE
        __shared__ int start_bp_fB;
        __shared__ int id_prev_fB;
        __shared__ int id_next_fB;
        __shared__ int circ_fB;
        __shared__ int or_fB;
        __shared__ int activ_fB;

        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            contig_fA = o_fragArray->id_c[id_fA];
            pos_fA = o_fragArray->pos[id_fA];
            sub_pos_fA = o_fragArray->sub_pos[id_fA]; // UPDATE
            l_cont_fA = o_fragArray->l_cont[id_fA];
            l_cont_bp_fA = o_fragArray->l_cont_bp[id_fA];
            sub_l_cont_fA = o_fragArray->sub_l_cont[id_fA];  // UPDATE
            len_bp_fA = o_fragArray->len_bp[id_fA];
            sub_len_fA = o_fragArray->sub_len[id_fA];  // UPDATE
            start_bp_fA = o_fragArray->start_bp[id_fA];
            id_prev_fA = o_fragArray->prev[id_fA];
            id_next_fA = o_fragArray->next[id_fA];
            circ_fA = o_fragArray->circ[id_fA];
            or_fA = o_fragArray->ori[id_fA];
            activ_fA = o_fragArray->activ[id_fA];

            contig_fB = o_fragArray->id_c[id_fB];
            pos_fB = o_fragArray->pos[id_fB];
            sub_pos_fB = o_fragArray->sub_pos[id_fB];  // UPDATE
            l_cont_fB = o_fragArray->l_cont[id_fB];
            l_cont_bp_fB = o_fragArray->l_cont_bp[id_fB];
            sub_l_cont_fB = o_fragArray->sub_l_cont[id_fB];  // UPDATE
            len_bp_fB = o_fragArray->len_bp[id_fB];
            sub_len_fB = o_fragArray->sub_len[id_fB];  // UPDATE
            start_bp_fB = o_fragArray->start_bp[id_fB];
            id_prev_fB = o_fragArray->prev[id_fB];
            id_next_fB = o_fragArray->next[id_fB];
            circ_fB = o_fragArray->circ[id_fB];
            or_fB = o_fragArray->ori[id_fB];
            activ_fB = o_fragArray->activ[id_fB];
        }
        __syncthreads();

        int contig_fi = o_fragArray->id_c[id_frag];
        int pos_fi = o_fragArray->pos[id_frag];
        int sub_pos_fi = o_fragArray->sub_pos[id_frag];  // UPDATE
        int l_cont_fi = o_fragArray->l_cont[id_frag];
        int l_cont_bp_fi = o_fragArray->l_cont_bp[id_frag];
        int sub_l_cont_fi = o_fragArray->sub_l_cont[id_frag];  // UPDATE
        int len_bp_fi = o_fragArray->len_bp[id_frag];
        int sub_len_fi = o_fragArray->sub_len[id_frag];  // UPDATE
        int circ_fi = o_fragArray->circ[id_frag];
        int id_prev_fi = o_fragArray->prev[id_frag];
        int id_next_fi = o_fragArray->next[id_frag];
        int start_bp_fi = o_fragArray->start_bp[id_frag];
        int or_fi = o_fragArray->ori[id_frag];
        int rep_fi = o_fragArray->rep[id_frag];
        int activ_fi = o_fragArray->activ[id_frag];
        int id_d_fi = o_fragArray->id_d[id_frag];

        if (id_frag < n_frags){
            if ( (activ_fA == 1) && ( activ_fB == 1) ){
                if (contig_fA != contig_fB){
                    if (contig_fi == contig_fA){
                        if (pos_fA == 0){
                            fragArray->pos[id_frag] = l_cont_fA - (pos_fi + 1);
                            fragArray->sub_pos[id_frag] = sub_l_cont_fA - (sub_pos_fi + sub_len_fi);  // UPDATE
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = l_cont_bp_fA - (start_bp_fi + len_bp_fi);
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi * -1;
                            if (pos_fi == l_cont_fA - 1){
                                fragArray->prev[id_frag] = -1;
                            }
                            else{
                                fragArray->prev[id_frag] = id_next_fi;
                            }
                            if (pos_fi == pos_fA){
                                fragArray->next[id_frag] = id_fB;
                            }
                            else{
                                fragArray->next[id_frag] = id_prev_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_fA + sub_l_cont_fB;  // UPDATE
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;


                        }
                        else{
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi;  // UPDATE
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->prev[id_frag] = id_prev_fi;
                            if (pos_fi == pos_fA){
                                fragArray->next[id_frag] = id_fB;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_fA + sub_l_cont_fB;  // UPDATE
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;

                        }
                    }
                    else if (contig_fi == contig_fB){
                        if (pos_fB == 0){
                            fragArray->pos[id_frag] = l_cont_fA + pos_fi;
                            fragArray->sub_pos[id_frag] = sub_l_cont_fA + sub_pos_fi;  // UPDATE
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = l_cont_bp_fA + start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_fB){
                                fragArray->prev[id_frag] = id_fA;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            fragArray->next[id_frag] = id_next_fi;
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_fA + sub_l_cont_fB;  // UPDATE
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                        else{
                            fragArray->pos[id_frag] = l_cont_fA + (l_cont_fB - (pos_fi + 1));
                            fragArray->sub_pos[id_frag] = sub_l_cont_fA + (sub_l_cont_fB - (sub_pos_fi + sub_len_fi));  // UPDATE
                            fragArray->id_c[id_frag] = contig_fA;
                            fragArray->start_bp[id_frag] = l_cont_bp_fA + (l_cont_bp_fB - (start_bp_fi + len_bp_fi));
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi;  // UPDATE
                            fragArray->circ[id_frag] = 0;
                            fragArray->id[id_frag] = id_frag;
                                fragArray->ori[id_frag] = or_fi * -1;
                            if (pos_fi == pos_fB){
                                fragArray->prev[id_frag] = id_fA;
                            }
                            else{
                                fragArray->prev[id_frag] = id_next_fi;
                            }
                            if (pos_fi == 0){
                                fragArray->next[id_frag] = -1;
                            }
                            else{
                                fragArray->next[id_frag] = id_prev_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA + l_cont_fB;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA + l_cont_bp_fB;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_fA + sub_l_cont_fB; // UPDATE
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;
                        }
                    }
                    else{
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                        fragArray->id_c[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi; // UPDATE
                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_fi;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_fi; // UPDATE
                        fragArray->id_d[id_frag] = id_d_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->rep[id_frag] = rep_fi;
                    }

                }
                else if (contig_fA == contig_fB){ // circular contig
                    if (contig_fi == contig_fA){
                        if ((pos_fA == 0) && (pos_fB == l_cont_fA - 1)){ //  creation of a circular contig !
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                            fragArray->id_c[id_frag] = contig_fi;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; // UPDATE
                            fragArray->circ[id_frag] = 1;
                            fragArray->ori[id_frag] = or_fi;
                            fragArray->id[id_frag] = id_frag;
                            if (pos_fi == pos_fA){
                                fragArray->prev[id_frag] = id_fB;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            if (pos_fi == l_cont_fA - 1){
                                fragArray->next[id_frag] = id_fA;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_fA; // UPDATE
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;

                        }
                        else if((pos_fA == l_cont_fA - 1) && (pos_fB == 0)){ //  creation of a circular contig !
                            fragArray->pos[id_frag] = pos_fi;
                            fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                            fragArray->id_c[id_frag] = contig_fi;
                            fragArray->start_bp[id_frag] = start_bp_fi;
                            fragArray->len_bp[id_frag] = len_bp_fi;
                            fragArray->sub_len[id_frag] = sub_len_fi; // UPDATE
                            fragArray->circ[id_frag] = 1;
                            fragArray->id[id_frag] = id_frag;
                            fragArray->ori[id_frag] = or_fi;
                            if (pos_fi == pos_fB){
                                fragArray->prev[id_frag] = id_fA;
                            }
                            else{
                                fragArray->prev[id_frag] = id_prev_fi;
                            }
                            if (pos_fi == l_cont_fA - 1){
                                fragArray->next[id_frag] = id_fB;
                            }
                            else{
                                fragArray->next[id_frag] = id_next_fi;
                            }
                            fragArray->l_cont[id_frag] = l_cont_fA;
                            fragArray->l_cont_bp[id_frag] = l_cont_bp_fA;
                            fragArray->sub_l_cont[id_frag] = sub_l_cont_fA; // UPDATE
                            fragArray->id_d[id_frag] = id_d_fi;
                            fragArray->activ[id_frag] = activ_fi;
                            fragArray->rep[id_frag] = rep_fi;

                        }
                    }
                    else{
                        fragArray->pos[id_frag] = pos_fi;
                        fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                        fragArray->id_c[id_frag] = contig_fi;
                        fragArray->start_bp[id_frag] = start_bp_fi;
                        fragArray->len_bp[id_frag] = len_bp_fi;
                        fragArray->sub_len[id_frag] = sub_len_fi; // UPDATE
                        fragArray->circ[id_frag] = circ_fi;
                        fragArray->id[id_frag] = id_frag;
                        fragArray->ori[id_frag] = or_fi;
                        fragArray->prev[id_frag] = id_prev_fi;
                        fragArray->next[id_frag] = id_next_fi;
                        fragArray->l_cont[id_frag] = l_cont_fi;
                        fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                        fragArray->sub_l_cont[id_frag] = sub_l_cont_fi; // UPDATE
                        fragArray->id_d[id_frag] = id_d_fi;
                        fragArray->activ[id_frag] = activ_fi;
                        fragArray->rep[id_frag] = rep_fi;
                    }

                }
            }
            else{
                fragArray->pos[id_frag] = pos_fi;
                fragArray->sub_pos[id_frag] = sub_pos_fi; // UPDATE
                fragArray->id_c[id_frag] = contig_fi;
                fragArray->start_bp[id_frag] = start_bp_fi;
                fragArray->len_bp[id_frag] = len_bp_fi;
                fragArray->sub_len[id_frag] = sub_len_fi; // UPDATE
                fragArray->circ[id_frag] = circ_fi;
                fragArray->id[id_frag] = id_frag;
                fragArray->ori[id_frag] = or_fi;
                fragArray->prev[id_frag] = id_prev_fi;
                fragArray->next[id_frag] = id_next_fi;
                fragArray->l_cont[id_frag] = l_cont_fi;
                fragArray->l_cont_bp[id_frag] = l_cont_bp_fi;
                fragArray->sub_l_cont[id_frag] = sub_l_cont_fi; // UPDATE
                fragArray->id_d[id_frag] = id_d_fi;
                fragArray->activ[id_frag] = activ_fi;
                fragArray->rep[id_frag] = rep_fi;
            }
        }
    }





     __global__ void fill_vect_dist(const float4* __restrict__ subFrags2Frags,
                                    frag* fragArray,
                                    float* sub_vect_dist,
                                    int* sub_vect_id_c,
                                    float* sub_vect_s_tot,
                                    int* sub_vect_pos,
                                    int* sub_vect_len,
                                    const int* __restrict__ collector_id,
                                    const int2* __restrict__ dispatcher,
                                    const int* __restrict__ sub_collector_id,
                                    const int2* __restrict__ sub_dispatcher,
                                    int n_sub_frags,
                                    int id_mut)
     {
        int2 dispatch_fi;
        int2 sub_dispatch_fi;
        int fi, pos_i, sub_len, sub_l_cont, sub_pos_i, frag_sub_pos,  or_fi, is_activ_fi, sub_fi, is_rep_fi, swap,s_tot;
        int contig_i;
        float s, dfi, fi_start_bp;
        float4 info_fi;
        int id_rep_fi, is_circle;
        int i = 0;
        int id_pix0 = blockIdx.x * blockDim.x + threadIdx.x;
        if (id_pix0 < n_sub_frags){
            info_fi = subFrags2Frags[id_pix0];
            dispatch_fi = dispatcher[(int) info_fi.x];
            sub_pos_i = (int) info_fi.w;
            sub_dispatch_fi = sub_dispatcher[id_pix0];
            i = 0;
            for(id_rep_fi = dispatch_fi.x; id_rep_fi < dispatch_fi.y; id_rep_fi ++){
                fi = collector_id[id_rep_fi];
                sub_fi = sub_collector_id[sub_dispatch_fi.x + i];
                is_activ_fi = fragArray->activ[fi];
                is_rep_fi = fragArray->rep[fi];
                contig_i = fragArray->id_c[fi];
                or_fi = fragArray->ori[fi];
                pos_i = fragArray->sub_pos[fi];
                sub_len = fragArray->sub_len[fi];
                sub_l_cont = fragArray->sub_l_cont[fi];
                is_circle = fragArray->circ[fi] == 1;
                s_tot = int2float(fragArray->circ[fi]) * int2float(fragArray->l_cont_bp[fi]) / 1000.0f;
                fi_start_bp =  int2float(fragArray->start_bp[fi]) ;
//                dfi = (or_fi == 1) * info_fi.y + (or_fi != 1) * info_fi.z ;
//                df_posi = (or_fi == 1) * ( pos_i + sub_pos_i) + (or_fi != 1) * (pos_i + sub_len - sub_pos_i);
                if (or_fi == 1){
                    dfi = info_fi.y;
                    frag_sub_pos = pos_i + sub_pos_i;
                }
                else{
                    dfi = info_fi.z;
                    frag_sub_pos = pos_i + sub_len - (sub_pos_i + 1);
                }
                sub_vect_dist[sub_fi * N_TMP_STRUCT + id_mut] = fi_start_bp / 1000.0f + dfi;
                sub_vect_id_c[sub_fi * N_TMP_STRUCT + id_mut] = contig_i;
                sub_vect_s_tot[sub_fi * N_TMP_STRUCT + id_mut] = s_tot;
                sub_vect_pos[sub_fi * N_TMP_STRUCT + id_mut] = frag_sub_pos;
                sub_vect_len[sub_fi * N_TMP_STRUCT + id_mut] = sub_l_cont;

                i += 1;
            }
        }
    }


     __global__ void uni_fill_vect_dist(const float4* __restrict__ subFrags2Frags,
                                             frag* fragArray,
                                             float* sub_vect_dist,
                                             int* sub_vect_id_c,
                                             float* sub_vect_s_tot,
                                             int* sub_vect_pos,
                                             int* sub_vect_len,
                                             const int* __restrict__ collector_id,
                                             const int2* __restrict__ dispatcher,
                                             const int* __restrict__ sub_collector_id,
                                             const int2* __restrict__ sub_dispatcher,
                                             int n_sub_frags)
     {
        int2 dispatch_fi;
        int2 sub_dispatch_fi;
        int fi, pos_i, sub_len, sub_pos_i, frag_sub_pos, or_fi, is_activ_fi, sub_fi, is_rep_fi, swap,s_tot;
        int contig_i;
        float s, dfi, fi_start_bp;
        float4 info_fi;
        int id_rep_fi, is_circle;
        int i = 0;
        int id_pix0 = blockIdx.x * blockDim.x + threadIdx.x;
        if (id_pix0 < n_sub_frags){
            info_fi = subFrags2Frags[id_pix0];
            dispatch_fi = dispatcher[(int) info_fi.x];
            sub_pos_i = (int) info_fi.w;
            sub_dispatch_fi = sub_dispatcher[id_pix0];
            i = 0;
            for(id_rep_fi = dispatch_fi.x; id_rep_fi < dispatch_fi.y; id_rep_fi ++){
                fi = collector_id[id_rep_fi];
                sub_fi = sub_collector_id[sub_dispatch_fi.x + i];
                is_activ_fi = fragArray->activ[fi];
                is_rep_fi = fragArray->rep[fi];
                contig_i = fragArray->id_c[fi];
                or_fi = fragArray->ori[fi];
                pos_i = fragArray->sub_pos[fi];
                sub_len = fragArray->sub_len[fi] - 1;
                is_circle = fragArray->circ[fi] == 1;
                s_tot = is_circle * int2float(fragArray->l_cont_bp[fi]) / 1000.0f;
                fi_start_bp =  int2float(fragArray->start_bp[fi]);
//                dfi = (or_fi == 1) * info_fi.y + (or_fi != 1) * info_fi.z;
                if (or_fi == 1){
                    dfi = info_fi.y;
                    frag_sub_pos = pos_i + sub_pos_i;
                }
                else{
                    dfi = info_fi.z;
                    frag_sub_pos = pos_i + sub_len - sub_pos_i;
                }
//                frag_sub_pos = (or_fi == 1) * ( pos_i + sub_pos_i) + (or_fi != 1) * (pos_i + sub_len - sub_pos_i);
                sub_vect_dist[sub_fi] = fi_start_bp / 1000.0f + dfi;
                sub_vect_id_c[sub_fi] = contig_i;
                sub_vect_s_tot[sub_fi] = s_tot;
                sub_vect_pos[sub_fi] = frag_sub_pos;
                sub_vect_len[sub_fi] = fragArray->sub_l_cont[fi];

                i += 1;
            }
        }
    }




    __inline__ __device__ double warpReduceSum(double val) {
      for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
      return val;
    }

    __inline__ __device__ double blockReduceSum(double val) {

      static __shared__ double shared[32]; // Shared mem for 32 partial sums
      int lane = threadIdx.x % warpSize;
      int wid = threadIdx.x / warpSize;

      val = warpReduceSum(val);     // Each warp performs partial reduction

      if (lane==0) shared[wid]=val;	// Write reduced value to shared memory

      __syncthreads();              // Wait for all partial reductions

      //read from shared memory only if that warp existed
      val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

      if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

      return val;
    }

    __global__ void eval_likelihood_on_zero(int* sub_vect_id_c,
                                            float* sub_vect_s_tot,
                                            int* sub_vect_pos,
                                            int* sub_vect_len,
                                            param_simu* P,
                                            float mean_size_frag,
                                            double* vect_likelihood,
                                            int* n_vals_intra,
                                            int n_frags)
    {

        __shared__ double sdata[1024]; // FERMI COMPATIBILITY
        int tid = threadIdx.x;
        int id = blockIdx.x * blockDim.x + tid;
        double val_expected = 0.0f;
        double val_likelihood = 0.0f;
        float s, s_tot, s_tot_z;
        int pos, tmp_len_cont, len_cont = 0;
        param_simu p = P[0];
        double n_tmp_vals = 0.0f;
        double tmp_likelihood = 0.0f;
        int id_frag = id;
        int condition = id_frag < n_frags;
        if (id_frag < n_frags){
//        for (int id_frag = id; id_frag < n_frags; id_frag += blockDim.x * gridDim.x){
            pos = sub_vect_pos[id_frag];
            len_cont = sub_vect_len[id_frag];
            s_tot = sub_vect_s_tot[id_frag];
            if (pos == 0){
                tmp_len_cont = len_cont * (len_cont - 1);
                atomicAdd(&n_vals_intra[0], tmp_len_cont / 2);
            }
            if (pos > 0){
                s = int2float(pos) * mean_size_frag;
                s_tot_z = int2float(len_cont) * mean_size_frag;
                if (s < p.d_max){
                    if (s_tot == 0){
                        val_expected = (double) rippe_contacts(s, p);
                    }
                    else{
                        val_expected = (double) rippe_contacts_circ(s, s_tot_z, p);
                    }
                }
                else{
                    val_expected = (double) p.v_inter;
                }
                n_tmp_vals = __int2double_rn(len_cont -  pos);
                tmp_likelihood = val_likelihood;
                val_likelihood =  tmp_likelihood - (val_expected * n_tmp_vals);
            }
        }

//        val_likelihood = blockReduceSum(val_likelihood); // KEPLER CODE !!!
        sdata[tid] = val_likelihood;
        __syncthreads();
        for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
            if(threadIdx.x < offset){
                // add a partial sum upstream to our own
                sdata[tid] += sdata[tid + offset];
            }
            // wait until all threads in the block have
            // updated their partial sums
            __syncthreads();
        }
        if ((tid == 0) && (condition ==1)){
            atomicAdd(&vect_likelihood[0], sdata[0]);
        }
    }

    __global__ void eval_all_likelihood_on_zero_1st(int* sub_vect_id_c,
                                                float* sub_vect_s_tot,
                                                int* sub_vect_pos,
                                                int* sub_vect_len,
                                                param_simu* P,
                                                float mean_size_frag,

                                                int* list_uniq_mutations,
                                                int* n_uniq_mutations,

                                                double* vect_likelihood,
                                                int* n_vals_intra,
                                                int n_frags)
    {
        __shared__ double sdata[1024]; // FERMI COMPATIBILITY
        int tid = threadIdx.x;
        int id = blockIdx.x * blockDim.x + tid;
        double val_expected = 0.0f;
        double tmp_likelihood = 0.0f;
        double val_likelihood[N_TMP_STRUCT] = {0.0f};
        double tmp_val = 0.0f;
        float s, s_tot, s_tot_z;
        int id_mut, k, pos, tmp_len_cont, len_cont = 0;
        param_simu p = P[0];
        double n_tmp_vals = 0.0;
        int condition = id < n_frags;
        for (int id_frag = id; id_frag < n_frags; id_frag += blockDim.x * gridDim.x){
            for (k = 0; k < n_uniq_mutations[0]; k ++){
                id_mut = list_uniq_mutations[k];
                pos = sub_vect_pos[id_frag * N_TMP_STRUCT + id_mut];
                len_cont = sub_vect_len[id_frag * N_TMP_STRUCT + id_mut];
                tmp_len_cont = len_cont * (len_cont -1);
                s_tot = sub_vect_s_tot[id_frag * N_TMP_STRUCT + id_mut];
                if (pos == 0){
                    atomicAdd(&n_vals_intra[id_mut], tmp_len_cont / 2);
                }
                if (pos > 0){
                    s = int2float(pos) * mean_size_frag;
                    s_tot_z = int2float(len_cont) * mean_size_frag;
                    if (s < p.d_max){
                        if (s_tot == 0){
                            val_expected = (double) rippe_contacts(s, p);
                        }
                        else{
                            val_expected = (double) rippe_contacts_circ(s, s_tot_z, p);
                        }
                    }
                    else{
                        val_expected = (double) p.v_inter;
                    }
                    n_tmp_vals = __int2double_rn(len_cont -  pos);
                    tmp_likelihood = val_likelihood[id_mut];
                    val_likelihood[id_mut] =  tmp_likelihood - val_expected * n_tmp_vals;
                }
            }
        }
//        for (id_mut=0; id_mut < 12; id_mut ++){ // KEPLER CODE !!
//            tmp_val = val_likelihood[id_mut];
//            __syncthreads();
//            tmp_val = blockReduceSum(tmp_val);
//            val_likelihood[id_mut] = tmp_val;
//        }
//        if (tid < 12){
//            atomicAdd(&vect_likelihood[tid], val_likelihood[tid]);
//        }

        for (k = 0; k < n_uniq_mutations[0]; k ++){
            id_mut = list_uniq_mutations[k];

            sdata[tid] = val_likelihood[id_mut];
            __syncthreads();
            for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
                if(threadIdx.x < offset){
                    // add a partial sum upstream to our own
                    sdata[tid] += sdata[tid + offset];
                }
                // wait until all threads in the block have updated their partial sums
                __syncthreads();
            }
            if ((tid == 0) && (condition == 1)){
                atomicAdd(&vect_likelihood[id_mut], sdata[0]);
            }
        }
    }


    __global__ void eval_all_likelihood_on_zero_2nd(int* list_uniq_mutations,
                                                    int* n_uniq_mutations,
                                                    param_simu* P,
                                                    double* vect_likelihood,
                                                    int* n_vals_intra,
                                                    double* n_tot_pxl)
    {
        int tid = threadIdx.x;
        int id = blockIdx.x * blockDim.x + tid;
        int id_mut;
        param_simu p = P[0];
        double intra_vals;
        double val_inter;
        double val_intra;
        double log_e =  0.43429448190325182f;
        if (tid < n_uniq_mutations[0]){
            id_mut = list_uniq_mutations[tid];
            intra_vals = __int2double_rn(n_vals_intra[id_mut]);
            val_inter = -1.0 * log_e *  (n_tot_pxl[0] - intra_vals) * p.v_inter;
            val_intra = vect_likelihood[id_mut] * log_e;
            vect_likelihood[id_mut] = val_intra + val_inter;
        }
    }

       __global__ void eval_all_scores(int* list_uniq_mutations,
                                       int* n_uniq_mutations,
                                       double* vect_likelihood_z,
                                       double* vect_likelihood_nz,
                                       double* curr_likelihood_nz_extract,
                                       double* curr_likelihood_nz,
                                       double* vect_all_score)
    {
        int tid = threadIdx.x;
        int id = blockIdx.x * blockDim.x + tid;
        int id_mut;
        if (id < n_uniq_mutations[0]){
            id_mut = list_uniq_mutations[tid];
            vect_all_score[id_mut] = vect_likelihood_nz[id_mut] +
                                     vect_likelihood_z[id_mut] +
                                     curr_likelihood_nz[0] - curr_likelihood_nz_extract[0];
        }
    }

     __global__ void prepare_sparse_call(const int* __restrict__ spData_row,
                                         int3* info_block,
                                         int* spData_block_csr,
                                         int *counter,
                                         int size_arr)
    {
        __shared__ int selec_smem[SIZE_BLOCK_4_SUB];
        __shared__ int counter_smem;
        int *counter_smem_ptr;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int condition = idx < size_arr;
        int start, id_next, next, curr;
        int3 info;
        int local_count = 0;

        if ((threadIdx.x == 0))
        {
            counter_smem_ptr = &counter_smem;
            counter_smem = 0;
            info.x = 0;
            info.y = 0;
            info.z = idx;
        }
        selec_smem[threadIdx.x] = -1;
        __syncthreads();
        if (condition == 1){
           // each counting thread writes its index to shared memory //
            id_next = min(idx + 1, size_arr - 1);
            curr = spData_row[idx];
            next = spData_row[id_next];
            if ((curr != next) || (threadIdx.x == 0) ||(threadIdx.x == SIZE_BLOCK_4_SUB - 1) || (idx == size_arr - 1) ){
                local_count = atomicAdd(counter_smem_ptr, 1);
                selec_smem[local_count] =  curr;
             }
        }
        __syncthreads();
        if (threadIdx.x == 0){
            local_count = counter_smem;
            counter_smem = atomicAdd(counter, counter_smem);
            info.x = local_count;
            info.y = counter_smem;
            info.z = idx;
            info_block[blockIdx.x] = info;
        }
        __syncthreads();
        if (selec_smem[threadIdx.x] >= 0){
            spData_block_csr[counter_smem + threadIdx.x] = selec_smem[threadIdx.x];
        }

    }

     __global__ void extract_sub_likelihood(const int* __restrict__ spData_dat,
                                            const int3* __restrict__ info_block,
                                            const int* __restrict__ spData_block_csr,
                                            const int* __restrict__ spData_row,
                                            const int* __restrict__ spData_col,
                                            param_simu* P,
                                            float mean_size_frag,
                                            float* sub_vect_pos_bp,
                                            int* sub_vect_id_c,
                                            float* sub_vect_s_tot,
                                            int* sub_vect_pos,
                                            int* sub_vect_len,

                                            double* vect_likelihood,
                                            int n_data,
                                            int n_sub_frags)
     {

        __shared__ double sdata[SIZE_BLOCK_4_SUB]; // FERMI CODE
        __shared__ float4 all_data_row[SIZE_BLOCK_4_SUB]; //
        __shared__ int list_fi[SIZE_BLOCK_4_SUB];
        __shared__ int3 param_block;

        int tid = threadIdx.x;
        int glob_id = blockIdx.x * blockDim.x + tid;
        double loc_likelihood = 0.0f; // 128 * 12

        int curr_fi, curr_fj, fi, fj, loc_fi;
        int local_id_i = 0;
        float si, sj, s_tot, s, sf, s_z, s_tot_z, pos_i, pos_j;
        int contig_i, contig_j;
        param_simu p = P[0];
        int i;
        double dat = 0.0f;
        double val_expected = 0.0f;
        double val_expected_z = 0.0f;
        double tmp_likelihood = 0.0f;

        if (tid == 0){
            param_block = info_block[blockIdx.x];
        }
        __syncthreads();

        int condition = (glob_id < n_data) ;

        if ((tid < param_block.x) && (condition == 1)){
            fi = spData_block_csr[param_block.y + tid];
            list_fi[tid] = fi;
            all_data_row[tid].x = int2float(sub_vect_id_c[fi]); // contig id
            all_data_row[tid].y = sub_vect_pos_bp[fi]; // kbp position
            all_data_row[tid].z = sub_vect_s_tot[fi]; // total kb length of the contigs
            all_data_row[tid].w = int2float(sub_vect_pos[fi]); // frag position
        }

        __syncthreads();

        if ((condition == 1)){
//            val_expected_z = 0.0f;
            curr_fi = spData_row[glob_id];
            for (i = 0; i < param_block.x; i ++){
                if ( curr_fi == list_fi[i]){
                    local_id_i = i;
                }
            }
            dat = (double) (spData_dat[glob_id]);
            curr_fj = spData_col[glob_id];

            fj = curr_fj;
            loc_fi = local_id_i;
            contig_i = __float2int_rd(all_data_row[loc_fi].x); // super stable ok
            contig_j = sub_vect_id_c[fj]; // ok !!


            s_tot = all_data_row[loc_fi].z;
            si = all_data_row[loc_fi].y; // stable ok
            pos_i = int2float(sub_vect_pos[curr_fi]);
            sj = sub_vect_pos_bp[fj]; // stable ...
            pos_j = int2float(sub_vect_pos[fj]);

            sf = si - sj;
            s = abs(sf);
            s_z = abs(pos_i - pos_j) * mean_size_frag;

            if (contig_i == contig_j){
                if (s_tot == 0){
                    val_expected = (double) rippe_contacts(s, p);
                    if (s_z < p.d_max){
                        val_expected_z = (double) rippe_contacts(s_z, p);
                    }
                    else{
                        val_expected_z = (double) p.v_inter;
                    }
                }
                else{
                    val_expected =  (double) rippe_contacts_circ(s, s_tot, p);
                    s_tot_z = int2float(sub_vect_len[fj]) * mean_size_frag;
                    if (s_z < p.d_max){
                        val_expected_z = (double) rippe_contacts_circ(s_z, s_tot_z, p);
                    }
                    else{
                        val_expected_z = (double) p.v_inter;
                    }
                }
            }
            else{
                val_expected = (double) p.v_inter;
                val_expected_z = (double) p.v_inter;
            }
            tmp_likelihood = evaluate_likelihood_pxl_double(val_expected, dat);
            loc_likelihood = tmp_likelihood + val_expected_z * 0.43429448190325182f;
//            loc_likelihood = tmp_likelihood;

        }
//        __syncthreads();
//        loc_likelihood = blockReduceSum(loc_likelihood); // KEPLER CODE
//        if (tid == 0){
//            atomicAdd(&vect_likelihood[0], loc_likelihood);
//        }

        sdata[tid] = loc_likelihood;
        __syncthreads();
        for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
            if(threadIdx.x < offset){
                // add a partial sum upstream to our own
                sdata[tid] += sdata[tid + offset];
            }
            // wait until all threads in the block have
            // updated their partial sums
            __syncthreads();
        }

        if ((tid == 0) && (condition == 1)){
            atomicAdd(&vect_likelihood[0], sdata[0]);
        }
    }


     __global__ void eval_sub_likelihood(const int* __restrict__ spData_dat,
                                         const int3* __restrict__ info_block,
                                         const int* __restrict__ spData_block_csr,
                                         const int* __restrict__ spData_row,
                                         const int* __restrict__ spData_col,
                                         param_simu* P,
                                         float mean_size_frag,
                                         const float* __restrict__ sub_vect_pos_bp,
                                         const int* __restrict__ sub_vect_id_c,
                                         const float* __restrict__ sub_vect_s_tot,
                                         const int* __restrict__ sub_vect_pos,
                                         const int* __restrict__ sub_vect_len,

                                         int* list_uniq_mutations,
                                         int* n_uniq_mutations,

                                         double* vect_likelihood,
                                         int n_data,
                                         int n_sub_frags)
     {
        __shared__ double loc_likelihood[N_STRUCT_BY_BLOCK_SIZE]; // SIZE_BLOCK_4_SUB * N_TMP_STRUCT
        __shared__ float4 all_data_row[N_STRUCT_BY_BLOCK_SIZE]    ; // SIZE_BLOCK_4_SUB * N_TMP_STRUCT
        __shared__ int list_fi[SIZE_BLOCK_4_SUB];
        __shared__ int3 param_block;
        int tid = threadIdx.x;
        int tidN_TMP_STRUCT = threadIdx.x * N_TMP_STRUCT;
        int glob_id = blockIdx.x * blockDim.x + tid;
        int curr_fi, curr_fj, fi, fiN_TMP_STRUCT, fj, loc_fi;
        int local_id_i = 0;
        float si, sj, s_tot, s, sf, s_z, s_tot_z, pos_i, pos_j;
        int contig_i, contig_j;
        param_simu p = P[0];
        int i, id_mut, k;
        double dat = 0.0f;
        double val_expected = 0.0f;
        double val_expected_z = 0.0f;
        double tmp_likelihood = 0.0f;

        if (tid == 0){
            param_block = info_block[blockIdx.x];
        }
        __syncthreads();

        int condition = (glob_id < n_data) ;

        if ((tid < param_block.x) && (condition == 1)){
            fi = spData_block_csr[param_block.y + tid];
            fiN_TMP_STRUCT = spData_block_csr[param_block.y + tid] * N_TMP_STRUCT;
            list_fi[tid] = fi;
            for (i = 0; i < N_TMP_STRUCT; i ++){

                all_data_row[tidN_TMP_STRUCT + i].x = int2float(sub_vect_id_c[fiN_TMP_STRUCT + i]); // contig id
                all_data_row[tidN_TMP_STRUCT + i].y = sub_vect_pos_bp[fiN_TMP_STRUCT + i]; // kbp position
                all_data_row[tidN_TMP_STRUCT + i].z = sub_vect_s_tot[fiN_TMP_STRUCT + i]; // total length of the contigs
                all_data_row[tidN_TMP_STRUCT + i].w = int2float(sub_vect_pos[fiN_TMP_STRUCT + i]); // frag position
            }
        }

        __syncthreads();

        if ((condition == 1)){
            curr_fi = spData_row[glob_id];
            for (i = 0; i < param_block.x; i ++){
                if ( curr_fi == list_fi[i]){
                    local_id_i = i * N_TMP_STRUCT;
                }
            }
            dat = (double) (spData_dat[glob_id]);
            curr_fj = spData_col[glob_id];
            for (k = 0; k < n_uniq_mutations[0]; k ++){
                id_mut = list_uniq_mutations[k];

                val_expected_z = 0.0f;
                fj = (curr_fj * N_TMP_STRUCT) + id_mut;
                loc_fi = local_id_i + id_mut;
                contig_i = __float2int_rd(all_data_row[loc_fi].x); // super stable ok
                contig_j = sub_vect_id_c[fj]; // ok !!


                s_tot = all_data_row[loc_fi].z;
                si = all_data_row[loc_fi].y; // stable ok
                pos_i = all_data_row[loc_fi].w;

                sj = sub_vect_pos_bp[fj]; // stable ...
                pos_j = int2float(sub_vect_pos[fj]);

                sf = si - sj;
                s = abs(sf);
                s_z = abs(pos_i - pos_j) * mean_size_frag;


                if (contig_i == contig_j){
                    if (s_tot == 0){
                        val_expected = (double) rippe_contacts(s, p);
                        if (s_z < p.d_max){
                            val_expected_z = (double) rippe_contacts(s_z, p);
                        }
                        else{
                            val_expected_z = (double) p.v_inter;
                        }
                    }
                    else{
                        val_expected =  (double) rippe_contacts_circ(s, s_tot, p);
                        if (s_z < p.d_max){
                            s_tot_z = int2float(sub_vect_len[fj]) * mean_size_frag;
                            val_expected_z = (double) rippe_contacts_circ(s_z, s_tot_z, p);
                        }
                        else{
                            val_expected_z = (double) p.v_inter;
                        }
                    }
                }
                else{
                    val_expected = (double)  p.v_inter;
                    val_expected_z = (double) p.v_inter;
                }
                tmp_likelihood = evaluate_likelihood_pxl_double(val_expected, dat);
                loc_likelihood[tidN_TMP_STRUCT + id_mut] = tmp_likelihood + val_expected_z * 0.43429448190325182f;
            }
        }
        else{
            for (id_mut = 0; id_mut < N_TMP_STRUCT; id_mut ++){
                loc_likelihood[tidN_TMP_STRUCT + id_mut] =  0.0f;
            }
        }
        __syncthreads();
        if ((tid < n_uniq_mutations[0]) && (condition == 1)){ // tid = id mutation ok
            tmp_likelihood = 0.0f;
            id_mut = list_uniq_mutations[tid];
            for (i = 0; i < SIZE_BLOCK_4_SUB; i+=1){
                tmp_likelihood += loc_likelihood[i * N_TMP_STRUCT + id_mut];
            }
            atomicAdd(&vect_likelihood[id_mut], tmp_likelihood);
        }
    }



     __global__ void evaluate_likelihood_sparse(const int* __restrict__ spData_dat,
                                                const int* __restrict__ spData_row,
                                                const int* __restrict__ spData_col,
                                                const int* __restrict__ id_single,

                                                param_simu* P,
                                                float mean_size_frag,

                                                float* sub_vect_pos_bp,
                                                int* sub_vect_id_c,
                                                float* sub_vect_s_tot,
                                                int* sub_vect_pos,
                                                int* sub_vect_len,

                                                double* vect_likelihood,

                                                int n_data_pxl,
                                                int n_frags)
     {
        __shared__ double sdata[1024];
        int tid = threadIdx.x;
        int id_pix0 = blockIdx.x * blockDim.x + tid;
        int fi, fj, pos_i, pos_j;
        float si, sj, s_tot, s, s_z, s_tot_z;
        int contig_i, contig_j;

        param_simu p = P[0];
        int row, col;
        int is_circle;

        double val_expected, val_expected_z, dat;
        double tmp_likelihood = 1.0;
        double loc_likelihood = 0.0f;
        int id_pix = id_pix0;

        for(id_pix = id_pix0; id_pix < n_data_pxl; id_pix += blockDim.x * gridDim.x){

            dat = (double) (spData_dat[id_pix]);

            fi = spData_row[id_pix];
//            fi = id_single[row];
            contig_i = sub_vect_id_c[fi];
            si = sub_vect_pos_bp[fi];
            s_tot = sub_vect_s_tot[fi];
            pos_i = sub_vect_pos[fi];

            fj = spData_col[id_pix];
//            fj = id_single[col];
            contig_j = sub_vect_id_c[fj];
            sj = sub_vect_pos_bp[fj];
            pos_j = sub_vect_pos[fj];

            s = abs(si - sj);
            s_z = abs(pos_i - pos_j) * mean_size_frag;
            s_tot_z = int2float(sub_vect_len[fi]) * mean_size_frag;

            if (contig_i == contig_j){
                if (s_tot == 0){
                    val_expected = (double) rippe_contacts(s, p);
                    if (s_z < p.d_max){
                        val_expected_z = (double) rippe_contacts(s_z, p);
                    }
                    else{
                        val_expected_z = (double) p.v_inter;
                    }
                }
                else{
                    val_expected =  (double) rippe_contacts_circ(s, s_tot, p);
                    if (s_z < p.d_max){
                        val_expected_z = (double) rippe_contacts_circ(s_z, s_tot_z, p);
                    }
                    else{
                        val_expected_z = (double) p.v_inter;
                    }
                }
//                if (s_z < p.d_max){
//                    val_expected_z = (double) rippe_contacts(s_z, p);
//                }
//                else{
//                    val_expected_z = (double) p.v_inter;
//                }
            }
            else{
                val_expected = (double) p.v_inter;
                val_expected_z = (double) p.v_inter;
            }
            // log10(dat|exp) - log10(0 | exp_z)

            tmp_likelihood = evaluate_likelihood_pxl_double(val_expected, dat) + val_expected_z * 0.43429448190325182f;
//            loc_likelihood += tmp_likelihood;
            loc_likelihood += tmp_likelihood;

        }
//        __syncthreads();
//        loc_likelihood = blockReduceSum(loc_likelihood); // KEPLER CODE
//        if (tid == 0){
//            atomicAdd(&vect_likelihood[0], loc_likelihood);
//        }

        sdata[tid] = loc_likelihood;
        __syncthreads();
        for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
            if(threadIdx.x < offset){
                // add a partial sum upstream to our own
                sdata[tid] += sdata[tid + offset];
            }
            // wait until all threads in the block have
            // updated their partial sums
            __syncthreads();
        }

        if (tid == 0){
            atomicAdd(&vect_likelihood[0], sdata[0]);
        }
    }



    __global__ void extract_uniq_mutations(frag* fragArray,
                                           int frag_a,
                                           int frag_b,
                                           int* list_uniq_mutations,
                                           int* list_valid_insert,
                                           int* n_uniq,
                                           int flip_eject)
    {
        int id_pix = threadIdx.x + blockDim.x * blockIdx.x;
        int len_ci, len_cj;
        int n;
        int start = 0;
        int j = 0;
        if (id_pix ==0 ){
            if (flip_eject == 1){
                list_uniq_mutations[0] = 0;
                list_uniq_mutations[1] = 1;
                list_uniq_mutations[2] = 2;
                list_uniq_mutations[3] = 3;
                start = 4;
                n = N_TMP_STRUCT;
            }
            else{
                list_uniq_mutations[0] = 2;
                list_uniq_mutations[1] = 3;
                start = 2;
                n = N_TMP_STRUCT - start;
            }
            len_ci = fragArray->l_cont[frag_a];
            len_cj = fragArray->l_cont[frag_b];
            if (len_cj == 1){
                n -= 4;
            }
            else{
                list_uniq_mutations[start + 0] = 4;
                list_uniq_mutations[start + 1] = 5;
                list_uniq_mutations[start + 2] = 6;
                list_uniq_mutations[start + 3] = 7;
                start += 4;
            }
            if (len_ci == 1){
                n -= 4;
            }
            else{
                list_uniq_mutations[start + 0] = 8;
                list_uniq_mutations[start + 1] = 9;
                list_uniq_mutations[start + 2] = 10;
                list_uniq_mutations[start + 3] = 11;
                start += 4;
            }
            for (int i=12; i<N_TMP_STRUCT; i++){
                if (list_valid_insert[i - 12] != -1){
                    list_uniq_mutations[start + j] = i;
                    j += 1;
                }
                else{
                    n -= 1;
                }
            }
            n_uniq[0] = n;
        }
    }



    __global__ void set_null(float* vect, int max_id)
    {
        int id_pix = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_pix < max_id){
            vect[id_pix] = 0.0;
        }
    }


    __global__ void copy_struct(frag* fragArray, frag* smplfragArray, int* id_contigs, int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        int id_c = 0;
        if (id_frag  < n_frags){
            fragArray->pos[id_frag] = smplfragArray->pos[id_frag];
            fragArray->sub_pos[id_frag] = smplfragArray->sub_pos[id_frag];
            id_c = smplfragArray->id_c[id_frag];
            fragArray->id_c[id_frag] = id_c;
            id_contigs[id_frag] = id_c;
            fragArray->circ[id_frag] = smplfragArray->circ[id_frag];
            fragArray->id[id_frag] = id_frag;
            fragArray->ori[id_frag] = smplfragArray->ori[id_frag];
            fragArray->start_bp[id_frag] = smplfragArray->start_bp[id_frag];
            fragArray->len_bp[id_frag] = smplfragArray->len_bp[id_frag];
            fragArray->sub_len[id_frag] = smplfragArray->sub_len[id_frag];
            fragArray->prev[id_frag] = smplfragArray->prev[id_frag];
            fragArray->next[id_frag] = smplfragArray->next[id_frag];
            fragArray->l_cont[id_frag] = smplfragArray->l_cont[id_frag];
            fragArray->sub_l_cont[id_frag] = smplfragArray->sub_l_cont[id_frag];
            fragArray->l_cont_bp[id_frag] = smplfragArray->l_cont_bp[id_frag];
            fragArray->rep[id_frag] = smplfragArray->rep[id_frag];
            fragArray->activ[id_frag] = smplfragArray->activ[id_frag];
            fragArray->id_d[id_frag] = smplfragArray->id_d[id_frag];
        }
    }

    __global__ void copy_gpu_array(double* dest, double* input, int max_id)
    {
        int id_pix_out = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_pix_out < max_id){
            for (int id_pix = id_pix_out; id_pix < max_id; id_pix += blockDim.x * gridDim.x){
                dest[id_pix] = input[id_pix];
            }
        }
    }


    __global__ void simple_copy(frag* fragArray, frag* smplfragArray, int n_frags)
    {
        int id_frag = threadIdx.x + blockDim.x * blockIdx.x;
        if (id_frag  < n_frags){
            fragArray->pos[id_frag] = smplfragArray->pos[id_frag];
            fragArray->sub_pos[id_frag] = smplfragArray->sub_pos[id_frag];
            fragArray->id_c[id_frag] = smplfragArray->id_c[id_frag];
            fragArray->circ[id_frag] = smplfragArray->circ[id_frag];
            fragArray->id[id_frag] = id_frag;
            fragArray->ori[id_frag] = smplfragArray->ori[id_frag];
            fragArray->start_bp[id_frag] = smplfragArray->start_bp[id_frag];
            fragArray->len_bp[id_frag] = smplfragArray->len_bp[id_frag];
            fragArray->sub_len[id_frag] = smplfragArray->sub_len[id_frag];
            fragArray->prev[id_frag] = smplfragArray->prev[id_frag];
            fragArray->next[id_frag] = smplfragArray->next[id_frag];
            fragArray->l_cont[id_frag] = smplfragArray->l_cont[id_frag];
            fragArray->sub_l_cont[id_frag] = smplfragArray->sub_l_cont[id_frag];
            fragArray->l_cont_bp[id_frag] = smplfragArray->l_cont_bp[id_frag];
            fragArray->rep[id_frag] = smplfragArray->rep[id_frag];
            fragArray->activ[id_frag] = smplfragArray->activ[id_frag];
            fragArray->id_d[id_frag] = smplfragArray->id_d[id_frag];
        }
    }




    __global__ void update_gpu_vect_frags(int* list_len,
                              frag* fragArray,
                              int * old_2_new_idx,
                              int* id_contigs,
                              float max_id,
                              int n_frags,
                              int* vect_min_id_c_new )
    {
        __shared__ float max_len;

        //get our index in the array
        int id_frag =  threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            max_len = __float2int_rd(list_len[0]);
        }
        __syncthreads();
        int min_id_c_new = vect_min_id_c_new[0];
        if (id_frag  < n_frags){
            int id_c = fragArray->id_c[id_frag];
            int id_c_new = max_id - old_2_new_idx[id_c];
            fragArray->id_c[id_frag] = id_c_new;
            id_contigs[id_frag] = id_c_new;
        }
    }


    __global__ void gl_update_pos(int* list_len,
                                  float4* pos,
                                  float4* color,
                                  float4* vel,
                                  float4* pos_gen,
                                  float4* vel_gen,
                                  frag* fragArray,
                                  int * old_2_new_idx,
                                  int* id_contigs,
                                  float max_id,
                                  int n_frags,
                                  int id_fi,
                                  int* vect_min_id_c_new,
                                  curandState* state,
                                  int n_rng,
                                  float dt)
    {
        __shared__ float max_len;

        //get our index in the array
        int id_frag =  threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x == 0){
            max_len = __float2int_rd(list_len[0]);
        }
        __syncthreads();
        float min_id_c_new = int2float(vect_min_id_c_new[0]);
        if (id_frag  < n_frags){

            int id_rng = id_frag % n_rng;
            float shift_y = (curand_normal(&state[id_rng]))*0.01;
            float shift_x = (curand_normal(&state[id_rng]))*0.01;
            float shift_z = (curand_normal(&state[id_rng]))*0.01;
            float shift_rot = shift_y;
            int id_c = fragArray->id_c[id_frag];
            int id_c_new = max_id - old_2_new_idx[id_c];
            fragArray->id_c[id_frag] = id_c_new;
            id_contigs[id_frag] = id_c_new;
            int is_circ = fragArray->circ[id_frag];
//            int is_circ = id_c_new % 2 == 0;
            float life = vel[id_frag].w;
            float pos_x;
            float l_cont;
            float radius;
            if (fragArray->l_cont[id_frag] > 1){

                pos_x = (int2float(fragArray->pos[id_frag]))/ max_len;
                if (is_circ == 1){
                    radius = (int2float(id_c_new - min_id_c_new) + shift_y * (id_frag == id_fi)) / (max_id-min_id_c_new) / 2 ;
                    l_cont = int2float(fragArray->l_cont[id_frag]) / max_len + 0.01f;
                    pos[id_frag].x = radius * 2;
                    pos[id_frag].y = 0 + radius * cos((pos_x + shift_rot) * 2 * M_PI  / l_cont ); // x plan coord
                    pos[id_frag].z = 0 + radius * sin((pos_x + shift_rot) * 2 * M_PI / l_cont ); // y plan coord;

                }
                else{
                    pos[id_frag].x = pos_x;
                    pos[id_frag].y = ((int2float(id_c_new) - min_id_c_new) + shift_y * (id_frag == id_fi)) / max(1.0f,(max_id-min_id_c_new));
    //                pos[id_frag].x = int2float(fragArray->pos[id_frag])/ max_len + 0.01f;
    //                pos[id_frag].y = (int2float(id_c_new - min_id_c_new) + shift_y * (id_frag == id_fi)) / (max_id-min_id_c_new) + 0.01f;
                    pos[id_frag].z = 0;

                }
                color[id_frag].w = 1.5;
            }
            else{
                float4 p = pos[id_frag];
                float4 v = vel[id_frag];
                life -= dt;
                if(life <= 0.f)
                {
                    p = pos_gen[id_frag];
                    v = vel_gen[id_frag];
                    life = 1.0f;
                }
                v.z -= 9.8f*dt;

                p.x += shift_y;;
                p.y += shift_z;
                p.z += shift_x;
                v.w = life;

                //update the arrays with our newly computed values
                pos[id_frag] = p;
                vel[id_frag] = v;

                //you can manipulate the color based on properties of the system
                //here we adjust the alpha
                color[id_frag].w = life;
            }
        }
    }


    __global__ void gpu_struct_2_pxl(frag* fragArray,
                                    int* pxl_frags,
                                    int* cumul_length,
                                    int max_id,
                                    float size_im_gl,
                                    int n_frags)
    {

        int id_frag =  threadIdx.x + blockDim.x * blockIdx.x;
        float pos, offset, tmp_pos;
        int id_c, pos_pix;

        if (id_frag < n_frags){
            id_c = max_id - fragArray->id_c[id_frag];
            pos = int2float(fragArray->pos[id_frag]);
            offset = int2float(cumul_length[id_c]);
            tmp_pos = (offset + pos) * size_im_gl / int2float(n_frags);
            pos_pix = min(__float2int_rd(tmp_pos), __float2int_rd(n_frags));
            pxl_frags[id_frag] = pos_pix;
        }
    }


//    __global__ void fill_im_zero(unsigned char* im_gl,
////                                 curandState* state, int n_rng,
//                                 int size_im_gl)
//    {
//        int pix_x =  threadIdx.x + blockDim.x * blockIdx.x;
//        int pix_y =  threadIdx.y + blockDim.y * blockIdx.y;
//        if ((pix_x < pix_y) && (pix_y < size_im_gl)){
////            int id_rng = (pix_x + pix_y) % n_rng;
////            float shift_y = curand_normal(&state[id_rng]) * 255;
////            unsigned char out = (unsigned char) shift_y;
//            int coord = pix_x *size_im_gl + pix_y;
//            im_gl[coord] = 0;
//        }
//    }


    __global__ void update_matrix(const int* __restrict__ spData_4GL_row,
                                  const int* __restrict__ spData_4GL_col,
                                  const int* __restrict__ spData_4GL_data,
                                  const int* __restrict__ spData_block_csr,
                                  const int3* __restrict__ info_block,
                                  int* pxl_frags,
                                  int* im_gl,
                                  int size_im_gl,
                                  int n_data)
    {
        __shared__ int sdata_contact[1024];
        __shared__ int sdata_coord[1024];
        __shared__ int3 param_block;
        int tid = threadIdx.x;
        int id_pix =  tid + blockDim.x * blockIdx.x;
        int condition = id_pix < n_data;
        int x, y, local_bin, i, fi, fj, coord;
        float val_cos_sin = (sqrtf(2)) / 2.0f;
        float size_im = int2float(size_im_gl);
        float mm = size_im * sqrtf(2.0f);
        if (tid == 0){
            param_block = info_block[blockIdx.x];
        }

        __syncthreads();

        if ((tid < param_block.x) && (condition == 1)){
            sdata_contact[tid] = 0;
            sdata_coord[tid] = spData_block_csr[param_block.y + tid];
        }

        __syncthreads();

        if (condition == 1){
            fi = pxl_frags[spData_4GL_row[id_pix]];
            fj = pxl_frags[spData_4GL_col[id_pix]];
            x = min(fi, fj);
            y = max(fi, fj);
//            coord = (size_im_gl - y - 1) * size_im_gl + x;
//            coord = (y + x - 1) * size_im_gl + (x - y + size_im_gl);

            /// coord image after rotation of pi/4
//            int i = __float2int_rd((y-mm/2)*val_cos_sin+(x-mm/2)*val_cos_sin+size_im/2);
//            int j = __float2int_rd(-(y-mm/2)*val_cos_sin+(x-mm/2)*val_cos_sin+size_im/2);

            coord = x * size_im_gl + y;

//            coord = i * size_im_gl + j;

            for (i = 0; i < param_block.x; i ++){
                if ( coord == sdata_coord[i]){
                    local_bin = i;
                }
            }
            sdata_contact[local_bin] += spData_4GL_data[id_pix];

        }
        __syncthreads();

        if ((tid < param_block.x) && (condition == 1)){
            atomicAdd(&im_gl[sdata_coord[tid]], sdata_contact[tid]);
        }
    }

    __global__ void update_gl_buffer(unsigned char* im_gl,
                                     int* im_cuda,
                                     int thresh,
                                     int id_max)
    {
        int pix_x = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned char out;
        float outf = 0.0f;
        if (pix_x < id_max){
            outf = (float) min(im_cuda[pix_x], thresh);
            out = (unsigned char) ( outf * 255/ (float) thresh );
            im_gl[pix_x] = out;
        }
    }


     __global__ void prepare_sparse_call_4_gl(const int* __restrict__ spData_row,
                                              const int* __restrict__ spData_col,
                                              int* spData_block_csr,
                                              const int* __restrict__ pxl_frags,
                                              int3* info_block,
                                              int *counter,
                                              int size_im_gl,
                                              int size_arr)
    {
        __shared__ int selec_smem[1024];
        __shared__ int counter_smem;
        int *counter_smem_ptr;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int condition = idx < size_arr;
        int condition_row, condition_col, condition_pix;
        int x, y, start, id_next, next_row, curr_row, next_col, curr_col, fi, next_fi, fj, next_fj, coord_pix;
        int3 info;
        int local_count = 0;
        float val_cos_sin = (sqrtf(2)) / 2.0f;
        float size_im = int2float(size_im_gl);
        float mm = size_im * sqrtf(2.0f);
        int max_coord = size_im_gl * size_im_gl - 1;

        if ((threadIdx.x == 0))
        {
            counter_smem_ptr = &counter_smem;
            counter_smem = 0;
            info.x = 0;
            info.y = 0;
            info.z = idx;
        }
        selec_smem[threadIdx.x] = -1;
        __syncthreads();
        if (condition == 1){
           // each counting thread writes its index to shared memory //
            id_next = min(idx + 1, size_arr - 1);
            curr_row = spData_row[idx];
            next_row = spData_row[id_next];
            curr_col = spData_col[idx];
            next_col = spData_col[id_next];
            fi = pxl_frags[curr_row];
            next_fi = pxl_frags[next_row];
            fj = pxl_frags[curr_col];
            next_fj = pxl_frags[next_col];

            condition_row = fi != next_fi;
            condition_col = fj != next_fj;

            x = min(fi, fj);
            y = max(fi, fj);
            coord_pix = x * size_im_gl + y; // coord image without rotation

            /// coord image after rotation of pi/4
//            y = min(fi, fj);
//            x = max(fi, fj);
//
//            int i = __float2int_rd((y-mm/2)*val_cos_sin+(x-mm/2)*val_cos_sin+size_im/2);
//            int j = __float2int_rd(-(y-mm/2)*val_cos_sin+(x-mm/2)*val_cos_sin+size_im/2);

//            coord = x * size_im_gl + y;

//            coord_pix = i * size_im_gl + j;
            condition_pix = (coord_pix >= 0) && ( coord_pix < max_coord);

//            if ((condition_pix) &&( (condition_row || condition_col) || (threadIdx.x == 0) || (threadIdx.x == 1023) || (idx == size_arr - 1) )){
            if (( (condition_row || condition_col) || (threadIdx.x == 0) || (threadIdx.x == 1023) || (idx == size_arr - 1) )){
                local_count = atomicAdd(counter_smem_ptr, 1);
//                selec_smem[local_count] =  coord_pix;
                selec_smem[local_count] =  coord_pix;
             }
        }
        __syncthreads();
        if ((threadIdx.x == 0) && (condition == 1)){
            local_count = counter_smem;
            counter_smem = atomicAdd(counter, counter_smem);
            info.x = local_count;
            info.y = counter_smem;
            info.z = idx;
            info_block[blockIdx.x] = info;
        }
        __syncthreads();
        if ((selec_smem[threadIdx.x] >= 0) && (condition == 1)){
            spData_block_csr[counter_smem + threadIdx.x] = selec_smem[threadIdx.x];
        }

    }



//    __global__ void csr_likelihood(const float* __restrict obsData2D,
//                                           frag* fragArray,
//                                           int* collector_id,
//                                           int2* dispatcher,
//                                           int4* id_sub_frags,
//                                           int4* rep_id_sub_frags,
//                                           float3* len_bp_sub_frags,
//                                           int3* accu_sub_frags,
//                                           double* likelihood,
//                                           param_simu* P,
//                                           int max_id_up_diag,
//                                           int max_id,
//                                           int n_bins,
//                                           int width_matrix,
//                                           float n_frags_per_bins)
//    {
//
//
//
//
//
//    }
} // extern "C"
