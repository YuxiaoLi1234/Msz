#include <iostream>
#include <float.h> 
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <parallel/algorithm>  
#include <unordered_map>
#include <random>
#include <iostream>
#include <cstring> 
#include <chrono> 
#include <cuda_runtime.h>
#include <string>
#include <omp.h>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <iomanip>
#include <chrono>

using std::count;
using std::cout;
using std::endl;

// nvcc -c kernel2.cu -o kernel2.o
// CUDA核函数，执行向量加法
// __device__ std::vector<float> decp_data;
__device__ float* decp_data ;
__device__ float* decp_data_copy ;
__device__ int directions1[36] =  {0,1,0,0,-1,0,1,0,0,-1,0,0,-1,1,0,1,-1,0,0,0, -1,  0,-1, 1, 0,0, 1,  0,1, -1,  -1,0, 1,   1, 0,-1};
__device__ int width;
__device__ int height;
__device__ int depth;
__device__ int num;
__device__ int* adjacency;
__device__ int* all_max; 
__device__ int* all_min;
__device__ int* unsigned_n;
__device__ int count_max;
__device__ int count_min;

__device__ int count_f_max;
__device__ int count_f_min;
__device__ int* maxi;
__device__ int* mini;
__device__ float bound;
__device__ int* or_maxi;
__device__ int* or_mini;
__device__ int* lowgradientindices;
__device__ float* input_data;
__device__ int* de_direction_as;
__device__ int* de_direction_ds;
__device__ int maxNeighbors = 12;

__device__ int direction_to_index_mapping[12][3] = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};   


__device__ int getDirection(int x, int y, int z){
    
    for (int i = 0; i < 12; ++i) {
        if (direction_to_index_mapping[i][0] == x && direction_to_index_mapping[i][1] == y && direction_to_index_mapping[i][2] == z) {
            return i+1;  
        }
    }
    return -1;  

// 26302898,3378820
// 27930227,32438238
}
__device__ int from_direction_to_index1(int cur, int direc){
    
    if (direc==-1) return cur;
    int row = cur % height;
    int rank1 = (cur / height) % width;
    int depth1 = cur/(width * height);
    // printf("%d %d\n", row, rank1);
    if (direc >= 1 && direc <= 12) {
        int delta_row = direction_to_index_mapping[direc-1][0];
        int delta_col = direction_to_index_mapping[direc-1][1];
        int delta_dep = direction_to_index_mapping[direc-1][2];
        
        
        int next_row = row + delta_row;
        int next_col = rank1 + delta_col;
        int next_dep = depth1 + delta_dep;
        // printf("%d \n", next_row * width + next_col);
        // return next_row * width + next_col + next_dep* (height * width);
        return next_row + next_col * height + next_dep* (height * width);
    }
    else {
        return -1;
    }
    // return 0;
};

__device__ void find_direction2 (int type, int index){
    float *data;
    int *direction_as;
    int *direction_ds;
    if(type==0){
        data = decp_data;
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
    }
    else{
        data = input_data;
        direction_as = or_maxi;
        direction_ds = or_mini;
    }
    
    float mini = 0;
    
    
    // std::vector<int> indexs = adjacency[index];
    int largetst_index = index;
    
    
        
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        if((data[i]>data[largetst_index] or (data[i]==data[largetst_index] and i>largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;
            // }
            
        };
    };
    int row_l = (largetst_index / (height)) % width;
    int row_i = (index / (height)) % width;
    
    int col_diff = row_l - row_i;
    int row_diff = (largetst_index % height) - (index % height);

    int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    direction_as[index] = getDirection(row_diff, col_diff,dep_diff);
    // if(index==8058 and type==0){
    //     printf("%d %d \n" ,direction_ds[index],or_mini[index]);
    // }
    
    

    mini = 0;
    largetst_index = index;
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        // if(i==8186 and index==8058 and type==0){
        //     printf("%.20f %.20f\n",data[i]-data[index],data[8057]-data[index]);
        //     // cout<<data[i]<<", "<<data[index]<<", "<<data[8057]<<endl;
        // }
        if((data[i]<data[largetst_index] or (data[i]==data[largetst_index] and i<largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;

            
        };
    };
    
    row_l = (largetst_index / (height)) % width;
    row_i = (index / (height)) % width;
    
    col_diff = row_l - row_i;
    row_diff = (largetst_index % height) - (index % height);

    dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    // row_l = (largetst_index % (height * width)) / width;
    // row_i = (index % (height * width)) / width;
    
    // row_diff = row_l - row_i;
    // col_diff = (largetst_index % width) - (index % width);

    // dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    
    direction_ds[index] = getDirection(row_diff, col_diff,dep_diff);
    
    
    
}
__global__ void find_direction (int type=0){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index>=num or lowgradientindices[index]==1){
        return;
    }
    
    float *data;
    int *direction_as;
    int *direction_ds;
    if(type==0){
        data = decp_data;
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
    }
    else{
        data = input_data;
        direction_as = or_maxi;
        direction_ds = or_mini;
    }
    
    float mini = 0;
    
    
    // std::vector<int> indexs = adjacency[index];
    int largetst_index = index;
    
    
        
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        if((data[i]>data[largetst_index] or (data[i]==data[largetst_index] and i>largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;
            // }
            
        };
    };
    // int row_l = (largetst_index % (height * width)) / width;
    // int row_i = (index % (height * width)) / width;
    
    // int row_diff = row_l - row_i;
    // int col_diff = (largetst_index % width) - (index % width);

    // int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    int row_l = (largetst_index / (height)) % width;
    int row_i = (index / (height)) % width;
    
    int col_diff = row_l - row_i;
    int row_diff = (largetst_index % height) - (index % height);

    int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    direction_as[index] = getDirection(row_diff, col_diff,dep_diff);
    
    // if(index==24654784 and type==0){
        
    //     printf("值：");
    //     printf("%d %d %d\n",row_diff, col_diff,dep_diff);
    //     printf("%d %d \n", largetst_index % 750, index % 750);
    //     // printf("%f %f \n" ,decp_data[index],input_data[index]);
    //     // for(int i=0;i<12;i++){
    //     //     int j = adjacency[index*12+i];
    //     //     if(j==-1){
    //     //         break;
    //     //     }
    //     //     printf("%f %f \n" ,decp_data[j],input_data[j]);
    //     // }
        
    // }
    
    

    mini = 0;
    largetst_index = index;
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        // if(i==8186 and index==8058 and type==0){
        //     printf("%.20f %.20f\n",data[i]-data[index],data[8057]-data[index]);
        //     // cout<<data[i]<<", "<<data[index]<<", "<<data[8057]<<endl;
        // }
        if((data[i]<data[largetst_index] or (data[i]==data[largetst_index] and i<largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;

            
        };
    };
    
    
    // row_l = (largetst_index % (height * width)) / width;
    // row_i = (index % (height * width)) / width;
    
    // row_diff = row_l - row_i;
    // col_diff = (largetst_index % width) - (index % width);

    // dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    row_l = (largetst_index / (height)) % width;
    row_i = (index / (height)) % width;
    
    col_diff = row_l - row_i;
    row_diff = (largetst_index % height) - (index % height);

    dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    direction_ds[index] = getDirection(row_diff, col_diff,dep_diff);
    
    
    
    
    
    return;

};
__global__ void checkElementKernel(int* array, int size, int target, bool* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        if (array[idx] == target) {
            *result = true;
        }
    }
}

__global__ void iscriticle(){
        
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(i>=num or lowgradientindices[i]==1){
            
            return;
        }
        
        bool is_maxima = true;
        bool is_minima = true;
        
        for (int index=0;index<12;index++) {
            int j = adjacency[i*12+index];
            if(j==-1){
                break;
            }
            if(lowgradientindices[j]==1){
                continue;
            }
            
                
            if (decp_data[j] > decp_data[i]) {
                
                is_maxima = false;
                
                break;
            }
            else if(decp_data[j] == decp_data[i] and j>i){
                is_maxima = false;
                break;
            }
        }
        for (int index=0;index< 12;index++) {
            int j = adjacency[i*12+index];
            if(j==-1){
                break;
            }
            if(lowgradientindices[j]==1){
                    continue;
            }
            
            if (decp_data[j] < decp_data[i]) {
                is_minima = false;
                break;
            }
            else if(decp_data[j] == decp_data[i] and j<i){
                is_minima = false;
                break;
            }
        }
        
        
        if((is_maxima && or_maxi[i]!=-1) or (!is_maxima && or_maxi[i]==-1)){
            int idx_fp_max = atomicAdd(&count_f_max, 1);
            // if(i==6345199){
            //     printf("%d %d \n",is_maxima,or_maxi[i]);
            // }
            all_max[idx_fp_max] = i;
            
        }
        
        else if ((is_minima && or_mini[i]!=-1) or (!is_minima && or_mini[i]==-1)) {
            int idx_fp_min = atomicAdd(&count_f_min, 1);// in one instruction
            
            all_min[idx_fp_min] = i;
            
        } 
        
       
        
}

__global__ void freeDeviceMemory() {
    // 释放 decp_data 指向的内存
    if (decp_data != nullptr) {
        delete[] decp_data;
        decp_data = nullptr;  // 避免野指针
    }
} 
__global__ void freeDeviceMemory1() {
    // 释放 decp_data 指向的内存
    if (de_direction_as != nullptr) {
        delete[] de_direction_as;
        de_direction_as = nullptr;  // 避免野指针
    }
    if (de_direction_ds != nullptr) {
        delete[] de_direction_ds;
        de_direction_ds = nullptr;  // 避免野指针
    }
}
__global__ void computeAdjacency() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num and lowgradientindices[i]==0) {
        
        int y = (i / (height)) % width; // Get the x coordinate
        int x = i % height; // Get the y coordinate
        int z = i / (width * height);
        int neighborIdx = 0;
        
        for (int d = 0; d < 12; d++) {
            
            int dirX = directions1[d * 3];     
            int dirY = directions1[d * 3 + 1]; 
            int dirZ = directions1[d * 3 + 2]; 
            int newX = x + dirX;
            int newY = y + dirY;
            int newZ = z + dirZ;
            int r = newX + newY * height + newZ* (height * width); // Calculate the index of the adjacent vertex
            // if(lowgradientindices[r]==1){
            //     continue;
            // }
            if (newX >= 0 && newX < height && newY >= 0 && newY < width && r < width*height*depth && newZ<depth && newZ>=0 && lowgradientindices[r]==0) {
                
                adjacency[i * maxNeighbors + neighborIdx] = r;
                neighborIdx++;

            }
        }

        // Fill the remaining slots with -1 or another placeholder value
        
        for (int j = neighborIdx; j < maxNeighbors; ++j) {
            adjacency[i * maxNeighbors + j] = -1;
        }
    }
}

__global__ void allocateDeviceMemory() {
    if (threadIdx.x == 0) { // 仅在一个线程上执行
        // printf("%d %d \n", threadIdx.x,num );
        all_max = new int[num];
        
        all_min = new int[num];
    }
    return;
}



// __global__ void fix_maxi_critical1(int direction){
//     int index_f = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (direction == 0 && index_f<count_f_max && lowgradientindices[all_max[index_f]]==0){
        
//         int index = all_max[index_f];
        
//         if (or_maxi[index]!=-1){
            
//             int next_vertex = from_direction_to_index1(index,or_maxi[index]);
            
//             int smallest_vertex = next_vertex;
//             float threshold = -DBL_MAX;
            
            
//             for(int j=0;j<12;j++){
//                 int i = adjacency[index*12+j];
//                 if(lowgradientindices[i]==1){
//                     continue;
//                 }
//                 if(i==-1){
//                     break;
//                 }
//                 if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
//                     smallest_vertex = i;
//                     threshold = input_data[i];
//                 }
//             }
            
//             threshold = decp_data[smallest_vertex];
//             // float diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
//             float d = (bound - (input_data[index]-decp_data[index]))/2.0;
            
//             if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
//                 de_direction_as[index]=or_maxi[index];
                
//                 return;
//             }
            
//             if(d>=1e-6){
                
//                 if(decp_data[index]==decp_data[next_vertex])
//                     {
                        
//                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
//                             d/=2;
//                         }
//                         if (abs(input_data[index]-decp_data[index]+d)<=bound){
//                             decp_data[index] -= d;
//                         }

                    
//                     }
//                 else{
//                     if(decp_data[index]>=decp_data[next_vertex]){
                        
//                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
//                                 d/=2;
//                         }
                        
//                         if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
//                             while(decp_data[index] - d < threshold and d>=2e-16)
//                             {
//                                 d/=2;
//                             }
                            
                            
//                         }
//                         else if(threshold>decp_data[next_vertex]){
                            
                            
//                             float diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/2;
                            
//                             if(diff2>1e-6){
//                                 while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
//                                     diff2/=2;
//                                 }
                                
//                                 if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
//                                     decp_data[smallest_vertex]-=diff2;
//                                     // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
//                                 }
                                
                                
//                             }
                            
//                         }

//                         if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex]){
//                             decp_data[index] -= d;
                            
//                         }
                        
                        
                   
//                 };
//                      }

                 
            
                
//             }
//             else{
                
//                 if(decp_data[index]>=decp_data[next_vertex]){
//                     if(abs(input_data[index]-(input_data[next_vertex] -bound+ decp_data[index])/2.0)<=bound){
//                         decp_data[index] = (input_data[next_vertex] -bound + decp_data[index])/2.0;
//                     }
//                     else{
                        
//                         decp_data[index] = input_data[index] - bound;
//                     }
                    
//                 }
                
//             }
            
            
        
//         }
//         else{
//             // printf("%d \n",or_maxi[index]);
//             int largest_index = from_direction_to_index1(index,de_direction_as[index]);
//             // float diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             float d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
//             if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
//                 de_direction_as[index] = -1;
//             }
//             if(d>=1e-6){
                
//                 if (decp_data[index]<=decp_data[largest_index]){
//                     if(abs(input_data[largest_index]-decp_data[index]+d)){
//                         decp_data[largest_index] = decp_data[index]-d;
//                     }
//                 }
                
            
                
//             }
            
//             else{
//                 if(decp_data[index]<=decp_data[largest_index]){
//                     decp_data[index] = input_data[index] + bound;
//                 }
                    
//             }
            
//         }
        
        
    
//     }
    
//     else if(direction == 1 && index_f<count_f_min && lowgradientindices[all_min[index_f]]==0){
//         int index = all_min[index_f];
//         if (or_mini[index]!=-1){
//             int next_vertex= from_direction_to_index1(index,or_mini[index]);
            
//             float diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
//             float d =  (bound-(input_data[index]-decp_data[index]))/2.0;
            
            
//             if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
//                 de_direction_ds[index]=or_mini[index];
//                 return;
//             }
            
//             if(diff>=1e-6 or d>=1e-6){
//                 if(decp_data[index]==decp_data[next_vertex]){
                    
                    
//                         while(abs(input_data[next_vertex]-decp_data[index]-d)>bound and d>=2e-16){
//                             d/=2;
//                         }
                        
//                         if(abs(input_data[index]-decp_data[index]-d)<=bound){
//                             decp_data[index]+=d;
//                         }
                    
                    
                    
                    
//                 }
//                 else{
//                     if(decp_data[index]<=decp_data[next_vertex]){
                        
//                             while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
//                                     diff/=2;
//                             }
                            
//                             if (abs(input_data[next_vertex]-decp_data[index]+d)<=bound and decp_data[index]<=decp_data[next_vertex]){
//                                 decp_data[next_vertex] = decp_data[index]-diff;
//                             }
                            
                            
                        
                        
//                 };

//                 }
                
                

                
//             }

//             else{
                
//                 if(decp_data[index]<=decp_data[next_vertex]){
//                     if(abs(input_data[index]-(input_data[next_vertex] + bound + decp_data[index])/2.0)<=bound){
//                         decp_data[index] = (input_data[next_vertex] + bound + decp_data[index])/2.0;
//                     }
//                     else{
//                         decp_data[index] = input_data[index] + bound;
//                     }
//                 }
//             }
            

            
            
            

            
        
//         }
    
//         else{
            
//             int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
//             float diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             // float d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            
//             if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
//                 de_direction_ds[index] = -1;
//                 return;
//             }
            
//             if (diff>=1e-6){
//                 if (decp_data[index]>=decp_data[largest_index]){
//                     while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
//                         diff/=2;
//                     }
                    
                    
//                     if(abs(input_data[index]-decp_data[index]+diff)<=bound){
//                         decp_data[index] -= diff;
//                     }
                    
                    
//                 }                    
//             }
            
                    
//             else{
//                 if (decp_data[index]>=decp_data[largest_index]){
//                     decp_data[index] = input_data[index] - bound;
//                 }   
    
//             }


               
//         }

        
//     }    
//     return;
// };
__global__ void fix_maxi_critical1(int direction){
    int index_f = blockIdx.x * blockDim.x + threadIdx.x;
    
    
        
    
    if (direction == 0 && index_f<count_f_max && lowgradientindices[all_max[index_f]]==0){
        
        int index = all_max[index_f];
      
	// printf("%d\n",index);
        if (or_maxi[index]!=-1){
            // printf("%d\n",index);
            // find_direction2(1,index);
            int next_vertex = from_direction_to_index1(index,or_maxi[index]);
            
            int smallest_vertex = next_vertex;
            float threshold = -DBL_MAX;
            
            
            for(int j=0;j<12;j++){
                int i = adjacency[index*12+j];
                if(i==-1){
                    break;
                }
                if(lowgradientindices[i]==1){
                    continue;
                }
                
                if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];
            // 对的
            // d是把index还要降低
            // 如果是tthresh的话，那它的下限就是：input_data[index]-(abs(inaput_data[index]-decp_data[index]))
            // 之前的
            // float diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            // float d = (decp_data[index] - input_data[index] + bound )/2.0;
            // // float d = (decp_data[index]-(input_data[index]-(abs(input_data[index]-decp_data_copy[index]))))/2.0;
            // float d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            // float diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            float diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            float d = (decp_data[index] - input_data[index] + bound )/2.0;
            float d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            float diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            // float diff = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // float d = (decp_data[index] - input_data[index] + (input_data[index]-decp_data_copy[index]))/2.0;
            // float d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // float diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            if(count_f_max==3 and count_f_min==0 and index==123716372){
                 printf("改变前");
                 printf("%d, %f\n", index, decp_data[index]);
                 printf("%d %f\n", next_vertex, decp_data[next_vertex]);
                 printf("%f %f \n",diff, d);
                 printf("%f %f \n",decp_data[index], decp_data[index]-d);
            //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
             }
            
            if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
                de_direction_as[index]=or_maxi[index];
            
                return;
            }
            
            if(d>=1e-6){
                
                if(decp_data[index]==decp_data[next_vertex])
                    {
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-6){
                            d/=2;
                        }
                        if (abs(input_data[index]-decp_data[index]+d)<=bound){
                            decp_data[index] -= d;
                        }
                    }
                else{
                    if(decp_data[index]>=decp_data[next_vertex]){
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-6){
                                d/=2;
                        }
                        
                        if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
                            while(decp_data[index] - d < threshold and d>=2e-6)
                            {
                                d/=2;
                            }
                            
                            
                        }
                        // else if(threshold>decp_data[next_vertex]){
                            
                            
                        //     float diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/64;
                            
                        //     if(diff2>=1e-6){
                        //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
                        //             diff2/=2;
                        //         }
                                
                        //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                        //             if(smallest_vertex==66783){cout<<"在这里11."<<endl;}
                        //             decp_data[smallest_vertex]-=diff2;
                        //             // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                        //         }
                                
                                
                        //     }
                            
                        // }

                        if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex] and d>=1e-6){
                            // if(index==1620477){
                            //     // cout<<"next_vertex: "<<decp_data[next_vertex]<<endl;
                            //     // cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<endl;
                            //     cout<<"before index: "<<decp_data[index]<<endl;
                                
                            // }
                            
                            decp_data[index] -= d;
                            
                            
                                            
                        }
                        // else if(abs(input_data[next_vertex]-(decp_data[next_vertex]+d1))<=bound and decp_data[index]>=decp_data[next_vertex] and d1>0){
                        //     // if(index==1620477){
                        //     //     // cout<<"next_vertex: "<<decp_data[next_vertex]<<endl;
                        //     //     // cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<endl;
                        //     //     cout<<"before index: "<<decp_data[index]<<endl;
                                
                        //     // }
                            
                        //     decp_data[next_vertex] += d1;
                            
                            
                                            
                        // }
                        
                        
                   
                };
                     }
            if(count_f_max==3 and count_f_min==0 and index==123716372){
                 printf("改变后");
                 printf("%d, %f\n", index, decp_data[index]);
                 printf("%d %f\n", next_vertex, decp_data[next_vertex]);
                 printf("%f %f \n",diff, d);
                 printf("%f %f \n",decp_data[index], decp_data[index]-d);
            //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
             }
                 
            
                
            }
            else{
                
                if(decp_data[index]>decp_data[next_vertex]){
                    
                    if(abs(input_data[index]-decp_data[next_vertex])<bound){
                            
                            float t = (decp_data[next_vertex]-(input_data[index]-bound))/2.0;
                            decp_data[index] = decp_data[next_vertex] - t;
                            // decp_data[next_vertex] = t;
                        }
                    else{
                        decp_data[index] = input_data[index] - bound;
                        
                    }
                    
                }
                else if(decp_data[index]==decp_data[next_vertex]){
                    // float bound1 = abs(input_data[index]-decp_data[index]);
                    //
                    float d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==157569){
                    //     cout<<"在这时候d: "<<d<<endl;
                    // }   
                    // float d = 1e-6;
                    if(abs(input_data[index]-decp_data[index]+d)<=bound){
                        
                        decp_data[index]-=d;
                    }
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]-d)<=bound){
                        // if(next_vertex==78){cout<<"在这里21"<<endl;}
                        decp_data[next_vertex]+=d;
                    }
                }
                
            }
            
            
        
        }
        else{
            // if(index==25026 and count_f_max<=770){
            //     cout<<"在这里"<<endl;
            // }
            // find_direction2(0,index);
            int largest_index = from_direction_to_index1(index,de_direction_as[index]);
            // 对的
            float diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            float d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // float diff = (input_data[index]-decp_data[index])/2.0;
            // float d = (input_data[largest_index]-decp_data[index])/2.0;
            // float d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // float diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // if(index==25026 and count_f_max<=770){
            //     cout<<"改变前"<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            //     cout<<or_direction_as[25026]<<de_direction_as[25026]<<endl;
            // }
            
            if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                de_direction_as[index] = -1;
            }
            if(d>=1e-6){
                
                if (decp_data[index]<=decp_data[largest_index]){
                    if(abs(input_data[largest_index]-decp_data[index]+d)){
                        // if(largest_index==66783){cout<<"在这里17"<<endl;}
                        decp_data[largest_index] = decp_data[index]-d;
                    }
                }
                
            
                
            }
            
            else{
                if(decp_data[index]<=decp_data[largest_index]){
                    // if(index==78){
                    //         cout<<"在这里1"<<endl;
                    //     }
                    decp_data[index] = input_data[index] + bound;
                }
                    
            }

            // if(index==15885 and count_f_max==7){
            //     cout<<"改变后"<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            // }
            
        }
        
        
    
    }
    
    else if (direction != 0 && index_f<count_f_min && lowgradientindices[all_min[index_f]]==0){
        int index = all_min[index_f];
        if (or_mini[index]!=-1){
            // find_direction2(1,index);
            int next_vertex= from_direction_to_index1(index,or_mini[index]);
            
            float diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            float d =  (bound+input_data[index]-decp_data[index])/2.0;
            // float d1 =  (bound-(input_data[next_vertex]-decp_data[next_vertex]))/2.0;
            
            float d1 = (decp_data[next_vertex]-input_data[next_vertex]+bound)/2.0;
            if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
                de_direction_ds[index]=or_mini[index];
                return;
            }

            // if(index == 6595 and count_f_min==5){
            //     cout<<"下降："<<endl;
            //     cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
            //     cout<<"diff: "<<diff<<endl;
            //     cout<<"d: "<<d<<endl;
            //     cout<<"d1: "<<d1<<endl;
            // }
            // if(count_f_max==0 and count_f_min==4){
            //      printf("改变前");
            //      printf("%d, %f\n", index, decp_data[index]);
            //      printf("%d %f\n", next_vertex, decp_data[next_vertex]);
            //      printf("%f %f \n",diff, d);
            //      printf("%f %f \n",decp_data[index], decp_data[index]-d);
            // //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            // //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
            // //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            // //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            //  }
            
            if(diff>=1e-6 or d>=1e-6 or d1>=1e-6){
                
                if(decp_data[index]==decp_data[next_vertex]){
                    
                      
                    
                        while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-6){
                            diff/=2;
                        }
                        
                        if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-6){
                            // if(index==344033 and count_f_min==2){cout<<"在这里22"<<d<<endl;}
                            decp_data[next_vertex]= decp_data[index]-diff;
                        }
                        else if(d1>=1e-6){
                            // if(index==344033 and count_f_min==2){cout<<"在这里23"<<d<<endl;}
                            decp_data[next_vertex]-=d1;
                        }
                        else if(d>=1e-6){
                            // if(index==344033 and count_f_min==2){cout<<"在这里24"<<d<<endl;}
                            decp_data[index]+=d;
                        }

                    
                    
                }
                else{
                    if(decp_data[index]<=decp_data[next_vertex]){
                        
                            while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-6){
                                    diff/=2;
                            }
                            
                            
                            if (abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and decp_data[index]<=decp_data[next_vertex] and diff>=1e-6){
                                // while(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff<1e-7){
                                //     diff*=2;
                                // }
                                // if(index==270808 and count_f_min==1){cout<<"在这里2！"<< endl;}
                                while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-6){
                                    diff*=2;
                                }
                                if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
                                    decp_data[next_vertex] = decp_data[index]-diff;
                                }
                                // if(index == 6595 and count_f_min==5){
                                //     cout<<"在这里1！"<< diff <<", "<<index<<", "<<decp_data[index]<<","<<input_data[index]<<","<<input_data[next_vertex]<<endl;

                                // }
                                // if(next_vertex==66783){cout<<"在这里13"<<endl;}
                                // decp_data[next_vertex] = decp_data[index]-diff;
                                // if(index==89797){
                                //         cout<<"在这里2"<<diff<<", "<<d<<endl;
                                // }

                                // decp_data[index]+=d;
                            }
                            // else if(abs(input_data[index]-decp_data[index]-d)<=bound and decp_data[index]<=decp_data[next_vertex] and d>0){
                            //     if(index==135569){cout<<"在这里23"<<endl;}
                            //     decp_data[index]+=d;
                            // }
                            else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-6){
                                while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-6){
                                    d1*=2;
                                }
                                // if(count_f_min<=12){cout<<"在这里2！"<<abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<<"," <<d1<< endl;}
                                if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and d1>=1e-6){
                                    decp_data[next_vertex]-=d1;
                                }
                                // else{
                                //     decp_data[index] += d;
                                // }
                                // else{
                                // decp_data[next_vertex] = input_data[next_vertex] - bound;}
                                
                            }
                            else{
                                decp_data[next_vertex] = input_data[next_vertex] - bound;
                                // if(index == 6595 and count_f_min==5){cout<<"在这里3！"<<abs(input_data[next_vertex]-bound-decp_data[next_vertex])<< endl;}
                            }
                            
                            
                        
                        
                };

                }
                
                

                
            }

            else{
                
                if(decp_data[index]<decp_data[next_vertex]){
                    // if(next_vertex==339928 and wrong_maxi_cp.size()==84){
                    //     cout<<"np下降："<<endl;
                    //     cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
                    //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
                    //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
                    //     cout<<"diff: "<<diff<<endl;
                    //     cout<<"d: "<<d<<endl;
                
                    //     }
                        
                        // if(abs(input_data[index]-(decp_data[next_vertex]))<=bound and abs(input_data[next_vertex]-decp_data[index])<=bound){
                        //     float t = decp_data[index];
                        //     decp_data[index] = decp_data[next_vertex];
                        //     if(next_vertex==66783){cout<<"在这里14"<<endl;}
                        //     decp_data[next_vertex] = t;
                            
                        // }
                        if(abs(input_data[next_vertex]-decp_data[index])<bound){
                            float t = (decp_data[index]-(input_data[index]-bound))/2.0;
                            // if(index==949999){cout<<"在这里24"<<endl;}
                            // decp_data[index] = decp_data[next_vertex];
                            // if(next_vertex==66783){cout<<"在这里14"<<endl;}
                            decp_data[next_vertex] = decp_data[index]-t;
                            
                        }
                        else{
                            // if(index==949999){cout<<"在这里29"<<endl;}
                            decp_data[index] = input_data[index] + bound;
                        }
                }
                
                else if(decp_data[index]==decp_data[next_vertex]){
                    float d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==949999){
                    //     cout<<"在这里99 "<<d<<endl;
                    // }   
                    // float d = 1e-6;
                    if(abs(input_data[index]-decp_data[index]-d)<=bound){
                        decp_data[index]+=d;
                    }
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
                        // if(next_vertex==66783){cout<<"在这里13"<<endl;}
                        decp_data[next_vertex]-=d;
                    }
                }
            }
            

            
            
            
        // if(index == 6595 and count_f_min==5){
        //         cout<<"下降后："<<endl;
        //         cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
        //         cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
        //         cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
        //         cout<<"diff: "<<diff<<endl;
        //         cout<<"d: "<<d<<endl;
        //         cout<<"d1: "<<d1<<endl;
        //         cout<<input_data[index]<<","<<input_data[next_vertex]<<endl;
        //     }
            
        
        }
    
        else{
            // find_direction2(0,index);
            int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
            float diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            float d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // if(count_f_min==84){
            //     cout<<"np下降："<<endl;
            //     cout<<"next: "<<largest_index<<", "<<decp_data[largest_index]<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[largest_index]<<endl;
            //     cout<<"diff: "<<diff<<endl;
            //     cout<<"d: "<<d<<endl;
                
            // }
            if(count_f_max==0 and count_f_min==4 and index==6655977){
                 printf("改变前");
                 printf("%d, %f\n", index, decp_data[index]);
                 printf("%d %f\n", largest_index, decp_data[largest_index]);
                 printf("%f %f \n",diff, d);
                 printf("%f %f \n",decp_data[index], decp_data[index]-d);
                 printf("%f %f \n",input_data[index], input_data[largest_index]);
            //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
             }
            if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
                de_direction_ds[index] = -1;
                return;
            }
            
            if (diff>=1e-6){
                if (decp_data[index]>=decp_data[largest_index]){
                    while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-6){
                        diff/=2;
                    }
                    
                    
                    if(abs(input_data[index]-decp_data[index]+diff)<=bound){
                        // if(index==999973){
                        //     cout<<"在这里2！"<<endl;
                        // }
                        
                        decp_data[index] -= diff;
                    }
                    
                    
                }                    
            }
            
                    
            else{
                if (decp_data[index]>=decp_data[largest_index]){
                    
                    // if(index==66783){cout<<"在这里15"<<endl;}
                    decp_data[index] = input_data[index] - bound;
                    if(count_f_max==0 and count_f_min==4 and index==6655977){
                        printf("改变后");
                        // printf("%d, %f\n", index, decp_data[index]);
                        printf("%d %f\n", largest_index, decp_data[largest_index]);
                        
                        // printf("%f %f \n",diff, d);
                        // printf("%f %f \n",decp_data[index], decp_data[index]-d);
                    //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
                    //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
                    //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
                    //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
                    }
                }   
    
            }


               
        }

        
    }    
    return;
}

__global__ void addKernel(int* globalVar) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("hello word from thefind gpu!\n");
    if(i<=20){
        atomicAdd(globalVar, 1);
    }
    
}


// void init_or_data(std::vector<int> *a, std::vector<int> *b, std::vector<int> *c, std::vector<int> *d, std::vector<float> *input_data1, std::vector<float> *decp_data1, int num){
    
//     int* temp;
    
//     int* temp1;
//     float* temp3;
//     int* tempd;
//     int* tempd1;
//     float* temp5;

//     cudaMalloc(&temp, num * sizeof(int));
//     cudaMalloc(&temp1, num * sizeof(int));
//     cudaMalloc(&tempd, num * sizeof(int));
//     cudaMalloc(&tempd1, num * sizeof(int));
//     cudaMalloc(&temp3, num * sizeof(float));
//     cudaMalloc((void**)&temp5, num * sizeof(float));
    



//     cudaMemcpy(temp, a->data(), num * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(temp1, b->data(), num * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(tempd, c->data(), num * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(tempd1, d->data(), num * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(temp3, input_data1->data(), num * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(temp5, decp_data1->data(), num * sizeof(float), cudaMemcpyHostToDevice);

//     cudaMemcpyToSymbol(or_maxi, &temp, sizeof(int*));
//     cudaMemcpyToSymbol(or_mini, &temp1, sizeof(int*));
//     cudaMemcpyToSymbol(de_direction_as, &tempd, sizeof(int*));
//     cudaMemcpyToSymbol(de_direction_ds, &tempd1, sizeof(int*));
//     cudaMemcpyToSymbol(input_data, &temp3, sizeof(float*));
//     cudaMemcpyToSymbol(decp_data, &temp5, sizeof(float*));
//     cudaDeviceSynchronize();
    

//     dim3 blockSize(1000);
    
//     dim3 gridSize((num + blockSize.x - 1) / blockSize.x);f
    
//     int* tempDevicePtr = nullptr;
//     size_t arraySize = num*6; // 确定所需的大小
//     cudaError_t cudaStatus = cudaMalloc(&tempDevicePtr, arraySize * sizeof(int));
    
//     cudaStatus = cudaMemcpyToSymbol(adjacency, &tempDevicePtr, sizeof(tempDevicePtr));
    
//     computeAdjacency<<<gridSize, blockSize>>>(num,100,100,6);

//     cudaDeviceSynchronize();
    


    
//     iscriticle<<<gridSize,blockSize>>>(num);

    
//     cudaDeviceSynchronize();

    
    
//     int host_count_f_max;
//     cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
//     int host_count_f_min;
//     cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
//     // cout<<host_count_f_max<<endl;
//     while(host_count_f_max>0 or host_count_f_min>0){
        
//         // cout<<host_count_f_max<<", "<<host_count_f_min<<endl;
//         dim3 blockSize1(1000);
//         dim3 gridSize1((host_count_f_max + blockSize1.x - 1) / blockSize1.x);
        
//         fix_maxi_critical1<<<gridSize1, blockSize1>>>(0);
//         cudaDeviceSynchronize();

//         dim3 blocknum(1000);
//         dim3 gridnum((host_count_f_min + blocknum.x - 1) / blocknum.x);
        
        
//         fix_maxi_critical1<<<gridnum, blocknum>>>(1);
//         cudaDeviceSynchronize();
//         // 重新检查错误cp个数
//         int initialValue = 0;
//         cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
//         if (cudaStatus != cudaSuccess) {
//             std::cerr << "cudaMemcpyToSymbol failed1: " << cudaGetErrorString(cudaStatus) << std::endl;
//         }
//         // int initialValue = 0;
//         cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
       
//         if (cudaStatus != cudaSuccess) {
//             std::cerr << "cudaMemcpyToSymbol failed2: " << cudaGetErrorString(cudaStatus) << std::endl;
//         }

//         iscriticle<<<gridSize, blockSize>>>(num);
        
//         cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
//         cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
//         // cout<<host_count_f_max<<", "<<host_count_f_min<<endl;
//         cudaDeviceSynchronize();
        
        
//     }
    
    
//     cudaStatus = cudaMemcpy(decp_data1->data(), temp5, num * sizeof(float), cudaMemcpyDeviceToHost);

//     if (cudaStatus != cudaSuccess) {
//             std::cerr << "cudaMemcpyToSymbol failed3: " << cudaGetErrorString(cudaStatus) << std::endl;
//     }
//     cudaDeviceSynchronize();
    
    
//     cudaFree(temp);
//     cudaFree(temp1);
//     cudaFree(temp3);
//     cudaFree(temp5);
    
    
    

    
//     return;
    
// }

void init_inputdata(std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<float> *input_data1,std::vector<float> *decp_data1,int width1, int height1, int depth1, std::vector<int> *low,float bound1){
    int* temp;
    
    int* temp1;
    int* d_data;
    
    

    float* temp3;
    float* temp4;
    
    int num1 = width1*height1*depth1;


    

    // cout<<num1<<endl;
    // size_t size = num1 * sizeof(int);
    cudaMemcpyToSymbol(width, &width1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(height, &height1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(depth, &depth1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num, &num1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(bound, &bound1, sizeof(float), 0, cudaMemcpyHostToDevice);

    cudaMalloc(&temp, num1 * sizeof(int));
    cudaMalloc(&temp1, num1 * sizeof(int));
    cudaMalloc(&temp3, num1 * 2 * sizeof(float));
    cudaMalloc(&temp4, num1 * 2 * sizeof(float));
    cudaMalloc(&d_data, num1 * sizeof(int));
    
    cudaMemcpy(temp3, input_data1->data(), num1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(temp4, decp_data1->data(), num1 * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_data, low->data(), num1 * sizeof(int), cudaMemcpyHostToDevice);

    int *d_temp;  // 用于在主机端临时存储设备内存地址
    size_t size = num1 * sizeof(int);

    // 为设备端数组分配内存
    cudaMalloc(&d_temp, size);

    // 将设备端内存地址复制到设备端全局指针
    cudaMemcpyToSymbol(all_max, &d_temp, sizeof(int*));
    cudaMemcpyToSymbol(lowgradientindices, &d_data, sizeof(int*));
    int *d_temp1;  // 用于在主机端临时存储设备内存地址
    size_t size1 = num1 * sizeof(int);

    // 为设备端数组分配内存
    cudaMalloc(&d_temp1, size1);

    // 将设备端内存地址复制到设备端全局指针
    cudaMemcpyToSymbol(all_min, &d_temp1, sizeof(int*));

    int *d_temp2;  // 用于在主机端临时存储设备内存地址
    size_t size4 = num1 * sizeof(int);
    // 为设备端数组分配内存
    cudaMalloc(&d_temp2, size4);

    // 将设备端内存地址复制到设备端全局指针
    cudaMemcpyToSymbol(de_direction_as, &d_temp2, sizeof(int*));

    int *d_temp3;  // 用于在主机端临时存储设备内存地址
    size_t size3 = num1 * sizeof(int);

    // 为设备端数组分配内存
    cudaMalloc(&d_temp3, size3);

    // 将设备端内存地址复制到设备端全局指针
    cudaMemcpyToSymbol(de_direction_ds, &d_temp3, sizeof(int*));
    
    cudaError_t cudaStatus = cudaMemcpyToSymbol(or_maxi, &temp, sizeof(int*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed80: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaStatus = cudaMemcpyToSymbol(or_mini, &temp1, sizeof(int*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed80: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaMemcpyToSymbol(input_data, &temp3, sizeof(float*));
    cudaMemcpyToSymbol(decp_data, &temp4, sizeof(float*));
    cudaDeviceSynchronize();
    

    dim3 blockSize(1024);
    
    dim3 gridSize((num1 + blockSize.x - 1) / blockSize.x);
    
    int* tempDevicePtr = nullptr;
    size_t arraySize = num1*12; // 确定所需的大小
    cudaStatus = cudaMalloc(&tempDevicePtr, arraySize * sizeof(int));
    
    cudaStatus = cudaMemcpyToSymbol(adjacency, &tempDevicePtr, sizeof(tempDevicePtr));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed80: " << cudaGetErrorString(cudaStatus) << std::endl;
        }

    computeAdjacency<<<gridSize, blockSize>>>();
    // cout<<"出发"<<endl;
    find_direction<<<gridSize, blockSize>>>(1);
    find_direction<<<gridSize, blockSize>>>();
    // cout<<"出发"<<endl;
    cudaDeviceSynchronize();
    
    cudaMemcpy(a->data(), temp, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b->data(), temp1, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c->data(), d_temp2, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d->data(), d_temp3, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaFree(temp);
    // cudaFree(temp1);
    // cudaFree(temp3);
    // cudaFree(tempDevicePtr);
    
    return;
}
__global__ void copyDeviceVarToDeviceMem(int *deviceMem,int *deviceMem1) {
    if (threadIdx.x == 0) {  // 只在一个线程上执行
        *deviceMem = *de_direction_as;
        *deviceMem1 = *de_direction_ds;
    }
}
__global__ void getlabel(int *label, int *un_sign_ds, int *un_sign_as, int type=0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int *direction_as;
    int *direction_ds;
    // if(type==0){
    //     printf("%d\n", i*2+1);
    // }
    if(i>=num or lowgradientindices[i]==1){
        return;
    }
    if(type==0){
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
    }
    else{
        direction_as = or_maxi;
        direction_ds = or_mini;
    }
    
    int cur = label[i*2+1];
    
    int next_vertex;
    
    if (cur!=-1 and direction_as[cur]!=-1){
        // printf("%d\n", cur);
        int direc = direction_as[cur];
        
        next_vertex = from_direction_to_index1(cur, direc);
        // switch (direc) {
        //     case 1:
        //         next_vertex = (row)*width + (rank1-1);
        //         break;
        //     case 2:
        //         next_vertex = (row-1)*width + (rank1);
        //         break;
        //     case 3:
        //         next_vertex = (row-1)*width + (rank1+1);
        //         break;
        //     case 4:
        //         next_vertex = (row)*width + (rank1+1);
        //         break;
        //     case 5:
        //         next_vertex = (row+1)*width + (rank1);
        //         break;
        //     case 6:
        //         next_vertex = (row+1)*width + (rank1-1);
        //         break;
        // };

        
        // 检查他的下一步是不是extrema以及他的下一步是否已经converge
        
        if (direction_as[next_vertex] != -1 and direction_as[label[next_vertex*2+1]] != -1){
            // if(i==7072){
            //     printf("%d %d %d %d %d\n",next_vertex);
            // }
            *un_sign_as+=1;
        }
        
        if(direction_as[i]!=-1){

            if(label[next_vertex*2+1]!=-1 and direction_as[label[next_vertex*2+1]]==-1){
                label[i*2+1] = label[next_vertex*2+1];
            }
            else{

                label[i*2+1] = next_vertex;
            }
            
            
        }
        else{
            label[i*2+1] = -1;
        };
        // if(i==66590){
        //     printf("%d %d %d %d %d\n",next_vertex,de_direction_as[next_vertex],de_direction_as[label[next_vertex*2+1]],label[next_vertex*2+1],label[i*2+1]);
        // }
        
    }

    
    
    cur = label[i*2];
    int next_vertex1;
    
    
    if (cur!=-1 and direction_ds[cur]!=-1){
        
        int direc = direction_ds[cur];
            
        // int row = (cur% (height * width)) / width;
        // int rank1 = cur%width;
        // int depth1 = cur/(width * height);
        
        next_vertex1 = from_direction_to_index1(cur, direc);

        
        // 下一步不是extrema或者下一步没有converge，检查下一步的label是不是extrema
        if (direction_ds[next_vertex1] != -1 and direction_ds[label[next_vertex1*2]] != -1){
            
            *un_sign_ds+=1;
            // printf("%d \n",de_direction_ds[label[next_vertex1*2]]);
            // printf("%d \n",i);
        }
        // 更新当前的点。
        // 检查当前点是不是extrema
        if(direction_ds[i]!=-1){
            // 如果不是extrema，如果下一步已经收敛，那就把标签替换为下一步的标签。
            // 首先要检查下一步是不是extrema
            if(label[next_vertex1*2]!=-1 and direction_ds[label[next_vertex1*2]]==-1){
                label[i*2] = label[next_vertex1*2];
            }
            else{
                label[i*2] = next_vertex1;
            }
            
            
        }
        else{
            label[i*2] = -1;
        };
        
        
    }

    

}

void fix_process(std::vector<int> *c,std::vector<int> *d,std::vector<float> *decp_data1){
    auto total_start2 = std::chrono::high_resolution_clock::now();
    int num1;
    cudaMemcpyFromSymbol(&num1, num, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    float* temp5;
    float time = 0.0;
    cudaError_t cudaStatus = cudaMalloc((void**)&temp5, num1 * sizeof(float));
    auto start2 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(temp5, decp_data1->data(), num1 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed7: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    time+=duration2.count();
    cudaStatus = cudaMemcpyToSymbol(decp_data, &temp5, sizeof(float*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed73: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
   
    
    
    
    

    cudaDeviceSynchronize();
    

    
    
    int* hostArray;
    cudaStatus = cudaMalloc((void**)&hostArray, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed70: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    
    // 从设备内存复制数据到主机内存
    cudaMemcpyToSymbol(de_direction_as, &hostArray, sizeof(int*));
    
    int* hostArray1;

    
    cudaStatus = cudaMalloc((void**)&hostArray1, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed71: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaStatus =  cudaMemcpyToSymbol(de_direction_ds, &hostArray1, sizeof(int*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed72: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    cudaDeviceSynchronize();
    

    dim3 blockSize(1024);
    
    dim3 gridSize((num1 + blockSize.x - 1) / blockSize.x);
    find_direction<<<gridSize,blockSize>>>();
    
    iscriticle<<<gridSize,blockSize>>>();
    // return;
    
    cudaDeviceSynchronize();

    
    
    int host_count_f_max;
    cudaStatus = cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    int host_count_f_min;
    cudaStatus = cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed12: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    // cout<<host_count_f_max<<", "<<host_count_f_min<<num1<<endl;
    // return;
    float elapsedTime;
    float wholeTime;
    float totalElapsedTime = 0.0;
    float totalElapsedTime_fcp = 0.0;
    float totalElapsedTime_fd = 0.0;
    cudaEvent_t start, stop;
    cudaEvent_t start1;
    cudaEventCreate(&start);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaEventRecord(start1, 0);
    // cout<<"wrong: "<<(host_count_f_max+host_count_f_min)/num1<<endl;
    while(host_count_f_max>0 or host_count_f_min>0){
        
        cout<<host_count_f_max<<", "<<host_count_f_min<<endl;

        
        dim3 blockSize1(1024);
        dim3 gridSize1((host_count_f_max + blockSize1.x - 1) / blockSize1.x);
        // cudaEventRecord(start, 0);
        fix_maxi_critical1<<<gridSize1, blockSize1>>>(0);
        // cudaDeviceSynchronize();
        // cudaDeviceSynchronize();

        dim3 blocknum(1024);
        dim3 gridnum((host_count_f_min + blocknum.x - 1) / blocknum.x);
        
        
        fix_maxi_critical1<<<gridnum, blocknum>>>(1);
        // cout<<"wanc"<<endl;
        cudaDeviceSynchronize();
        
        // 重新检查错误cp个数
        int initialValue = 0;
        cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        // if (cudaStatus != cudaSuccess) {
        //     std::cerr << "cudaMemcpyToSymbol failed4: " << cudaGetErrorString(cudaStatus) << std::endl;
        // }
        // int initialValue = 0;
        cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));

        // if (cudaStatus != cudaSuccess) {
        //     std::cerr << "cudaMemcpyToSymbol failed5: " << cudaGetErrorString(cudaStatus) << std::endl;
        // }
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);

        // 计算这次迭代的时间并加到总时间上
        // cudaEventElapsedTime(&elapsedTime, start, stop);
        // totalElapsedTime+=elapsedTime;
        // std::cout << "Average Time Per Iteration = " << elapsedTime << " ms" << std::endl;
        // cudaEventRecord(start, 0);

        iscriticle<<<gridSize, blockSize>>>();
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);

        // 计算这次迭代的时间并加到总时间上
        // cudaEventElapsedTime(&elapsedTime, start, stop);
        // totalElapsedTime_fcp+=elapsedTime;

        // cudaEventRecord(start, 0);
        find_direction<<<gridSize,blockSize>>>();
        
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);
        // totalElapsedTime_fd+=elapsedTime;
        // 计算这次迭代的时间并加到总时间上
        
        
        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // cout<<host_count_f_max<<", "<<host_count_f_min<<endl;
        cudaDeviceSynchronize();
        
        // exit(0);
    }
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    
    // cudaEventRecord(start, 0);
    find_direction<<<gridSize,blockSize>>>();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // totalElapsedTime_fd+=elapsedTime;
    // cudaEventElapsedTime(&wholeTime, start1, stop);
    // cout<<"["<<totalElapsedTime/wholeTime<<", "<<totalElapsedTime_fcp/wholeTime<<", "<<totalElapsedTime_fd/wholeTime<<"],"<<endl;;
    start2 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(decp_data1->data(), temp5, num1 * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaDeviceSynchronize();

    


    

    // 从设备内存复制数据到主机内存
    // cudaMemcpy(hostArray1, de_direction_ds, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c->data(), hostArray, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d->data(), hostArray1, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end2 = std::chrono::high_resolution_clock::now();

    // Calculate duration
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    time+=duration2.count();
    // delete[] hostArray;
    // delete[] hostArray1;
    // delete[] temp5;
    cudaFree(temp5);
    cudaFree(hostArray);
    cudaFree(hostArray1);
    // cudaFree(num1);
    
    end2 = std::chrono::high_resolution_clock::now();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - total_start2);
    // printf("%f, ",time/duration2.count());
    

    return;
    
}

__global__ void copyDeviceToArray(int* hostArray,int* hostArray1) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num) {
        
        hostArray[index] = de_direction_as[index];
        
        hostArray1[index] = de_direction_ds[index];
    }
    
}


__global__ void initializeWithIndex(int* label, int size, int type=0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        int *direction_ds;
        int *direction_as;
        if(type==0){
            direction_ds = de_direction_ds;
            direction_as = de_direction_as;
        }
        else{
            
            direction_ds = or_mini;
            direction_as = or_maxi;
        
        }

        if(direction_ds[index]!=-1){
            label[index*2] = index;
        }
        else{
            label[index*2] = -1;
        }

        if(direction_as[index]!=-1){
            label[index*2+1] = index;
        }
        else{
            label[index*2+1] = -1;
        }
    }
}

void mappath1(std::vector<int> *label, std::vector<int> *direction_as, std::vector<int> *direction_ds, int type=0){
    int num1;
    cudaMemcpyFromSymbol(&num1, num, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    int *un_sign_as;
    cudaMalloc((void**)&un_sign_as, sizeof(int));
    cudaMemset(un_sign_as, 0, sizeof(int));

    int *un_sign_ds;
    cudaMalloc((void**)&un_sign_ds, sizeof(int));
    cudaMemset(un_sign_ds, 0, sizeof(int));

    
    
    
    dim3 blockSize1(1024);
    dim3 gridSize1((num1 + blockSize1.x - 1) / blockSize1.x);

    int* label_temp;
    cudaError_t cudaStatus = cudaMalloc((void**)&label_temp, num1*2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed60: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    
    
    int h_un_sign_as = num1;
    int h_un_sign_ds = num1;
    // int *un_sign_as = 0;
    // int *un_sign_ds = 0;
    int* hostArray;
    cudaStatus = cudaMalloc((void**)&hostArray, num1 * sizeof(int));
    // cout<<num1<<"大小"<<endl;
    // cudaMemcpy(decp_data1->data(), temp5, num1 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaStatus = cudaMemcpy(hostArray,direction_as->data(), num1 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed76: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    int* hostArray1;
    cudaStatus = cudaMalloc((void**)&hostArray1, num1 * sizeof(int));
    cudaStatus = cudaMemcpy(hostArray1,direction_ds->data(),  num1 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed78: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    if(type==0){
        // cout<<"集哪里"<<endl;
        
        // 从设备内存复制数据到主机内存
        
        cudaMemcpyToSymbol(de_direction_as, &hostArray, sizeof(int*));
        
        
        cudaStatus =  cudaMemcpyToSymbol(de_direction_ds, &hostArray1, sizeof(int*));
        if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpyToSymbol failed72: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
        
        
    }
    // else{
    //     cudaMemcpyToSymbol(or_maxi, &hostArray, sizeof(int*));
    //     cudaMemcpyToSymbol(or_mini, &hostArray1, sizeof(int*));
    // }
    cudaDeviceSynchronize();
    initializeWithIndex<<<gridSize1, blockSize1>>>(label_temp, num1,type);
    cudaDeviceSynchronize();
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        // cout<<h_un_sign_as<<","<<h_un_sign_ds<<endl;
        int zero = 0;
        int zero1 = 0;
        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize1,blockSize1>>>(label_temp,un_sign_as,un_sign_ds, type);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
        // exit(0);
    }   

    cudaDeviceSynchronize();

    
    cudaStatus = cudaMemcpy(label->data(), label_temp, num1 *2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed61: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    if(type==0){
        cudaFree(label_temp);
    }
    
    cudaFree(hostArray1);
    cudaFree(hostArray);
    return;
};
