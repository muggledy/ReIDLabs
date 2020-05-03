#%%cython --cplus --annotate

'''
用于"基于区域的对比度方法的显著区域检测(RC)"
Author: SongpingWang (https://blog.csdn.net/wsp_1138886114)
'''

import numpy as np
import cv2
cimport cython
cimport numpy as np
from cpython cimport array
from cython.parallel import prange
from libc.math cimport pow
from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport fabs
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from scipy.spatial.distance import pdist,squareform

###基于区域的对比度—图像分割的区域对比度计算###
#https://blog.csdn.net/wsp_1138886114/article/details/103717093

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Build_Regions_Contrast(int regNum, np.ndarray[np.int32_t, ndim=2] regIdx1i,
                            int[:,:] colorIdx1i, float[:,:,:]color3fv, float sigmaDist,float ratio,float thr):
    cdef:
        int height = regIdx1i.shape[0]
        int width = regIdx1i.shape[1]
        int colorNum = color3fv.shape[1]
        float cx = <float>width / 2.0
        float cy = <float>height / 2.0
        Py_ssize_t x = 0, y = 0, i=0, j=0, m=0, n=0, yi=0, xi=0,iii=0

    pixNum_np = np.zeros(regNum,dtype=np.int64)
    pixNum_np = np.bincount(regIdx1i.reshape(1, width * height)[0])
    
    ybNo_np = np.zeros(regNum, dtype=np.float64)
    regs_mX = np.zeros(regNum, dtype=np.float64)
    regs_mY = np.zeros(regNum, dtype=np.float64)
    regs_vX = np.zeros(regNum, dtype=np.float64)
    regs_vY = np.zeros(regNum, dtype=np.float64)
    
    freIdx_f64 = np.zeros((regNum, colorNum), dtype=np.float64)
    regColor_np = np.zeros((regNum, colorNum), dtype=np.int32)
    tile_pixNum = np.zeros((regNum, colorNum), dtype=np.int64)
    regs_np = np.zeros((regNum, 4), dtype=np.float64)
    
    cdef int [:,::1]regColor_view = regColor_np
    cdef double[:,::1]regs_view = regs_np

    with nogil:
        for y in range(height):
            for x in range(width):
                regs_view[regIdx1i[y, x], 0] = fabs(x - cx)  # ad2c_0
                regs_view[regIdx1i[y, x], 1] = fabs(y - cy)  # ad2c_1
                regs_view[regIdx1i[y, x], 2] += x            # region center x coordinate
                regs_view[regIdx1i[y, x], 3] += y            # region center y coordinate
                regColor_view[regIdx1i[y, x], colorIdx1i[y, x]] += 1

    regs_np[:, 0] = np.divide(regs_np[:, 0], pixNum_np * width)
    regs_np[:, 1] = np.divide(regs_np[:, 1], pixNum_np * height)
    
    regs_mX = np.divide(regs_np[:, 2], pixNum_np)
    regs_mY = np.divide(regs_np[:, 3], pixNum_np)
    
    regs_np[:, 2] = np.divide(regs_mX, width)
    regs_np[:, 3] = np.divide(regs_mY, height)

    freIdx_f64 = regColor_np.astype(np.float64)
    tile_pixNum = np.tile(pixNum_np[:, np.newaxis], (1, colorNum))
    freIdx_f64 = np.divide(freIdx_f64, tile_pixNum)

    similar_dist = np.zeros((colorNum, colorNum), dtype=np.float64)
    similar_dist = squareform(pdist(color3fv[0]))

    rDist_np = np.zeros((regNum, regNum), np.float64)
    regSal1d_np = np.zeros((1, regNum), np.float64)
    
    cdef double[:,:]regs_view2 = regs_np
    cdef double[:]mX_view = regs_mX
    cdef double[:]mY_view = regs_mY
    cdef double[::1]vX_view = regs_vX
    cdef double[::1]vY_view = regs_vY
    
    cdef double[:,:]similar_dist_view = similar_dist
    cdef double[:,::1]rDist_view = rDist_np
    cdef double[:,::1]regSal1d_view = regSal1d_np
    cdef double[:,:]freIdx_f64_view = freIdx_f64
    cdef long long[:]pixNum_view = pixNum_np
    cdef double dd_np = 0.0

    with nogil:
        for i in range(regNum):
            for j in range(regNum):
                if i < j:
                    for m in range(colorNum):
                        for n in range(colorNum):
                            if freIdx_f64_view[j, n] != 0.0 and freIdx_f64_view[i, m] != 0.0:
                                dd_np += similar_dist_view[m,n] * freIdx_f64_view[i, m] * freIdx_f64_view[j, n]
                    rDist_view[i][j] = dd_np * exp(-1.0 * (pow((regs_view2[i, 2]-regs_view2[j, 2]),2)+pow((regs_view2[i, 3]-regs_view2[j, 3]),2)) / sigmaDist)
                    rDist_view[j][i] = rDist_view[i][j]
                    dd_np = 0.0
                regSal1d_view[0, i] += pixNum_view[j] * rDist_view[i, j]
            regSal1d_view[0, i] *= exp(-9.0 * (pow(regs_view2[i, 0],2) + pow(regs_view2[i, 1],2)))
    
        for yi in range(height):
            for xi in range(width):
                vX_view[regIdx1i[yi, xi]] += fabs(xi - mX_view[regIdx1i[yi, xi]])
                vY_view[regIdx1i[yi, xi]] += fabs(yi - mY_view[regIdx1i[yi, xi]])
    regs_vX = np.divide(regs_vX,pixNum_np)
    regs_vY = np.divide(regs_vY,pixNum_np)
    
    #在x和y边界区域的边界像素的数量
    cdef:
        vector[int] bPnts0
        vector[int] bPnts1
        array.array pnt = array.array('i',[0,0])
        int [:] pnt_view = pnt
        int wGap = <int>(width * ratio + 0.5)
        int hGap = <int>(height * ratio + 0.5)
        int sx = 0, sx_right = width - wGap
        double xR = 0.25* hGap
        double yR = 0.25* wGap
        double [::1] ybNum = ybNo_np

    with nogil:
        # top region
        while pnt_view[1] != hGap:
            pnt_view[0] = sx
            while pnt_view[0] != width:
                ybNum[regIdx1i[pnt_view[1], pnt_view[0]]] += 1
                bPnts0.push_back(pnt_view[0])
                bPnts1.push_back(pnt_view[1])
                pnt_view[0] += 1
            pnt_view[1] += 1

        pnt_view[0] = 0
        pnt_view[1] = height - hGap
        # Bottom region
        while pnt_view[1] != height:
            pnt_view[0] = sx
            while pnt_view[0] != width:
                ybNum[regIdx1i[pnt_view[1], pnt_view[0]]] += 1
                bPnts0.push_back(pnt_view[0])
                bPnts1.push_back(pnt_view[1])
                pnt_view[0] += 1
            pnt_view[1] += 1

        pnt_view[0] = 0
        pnt_view[1] = 0
        # Left Region
        while pnt_view[1] != height:
            pnt_view[0] = sx
            while pnt_view[0] != wGap:
                ybNum[regIdx1i[pnt_view[1], pnt_view[0]]] += 1
                bPnts0.push_back(pnt_view[0])
                bPnts1.push_back(pnt_view[1])
                pnt_view[0] += 1
            pnt_view[1] += 1

        pnt_view[0] = sx_right
        pnt_view[1] = 0
        # Right Region
        while pnt_view[1] != height:
            pnt_view[0] = sx_right
            while pnt_view[0] != width:
                ybNum[regIdx1i[pnt_view[1], pnt_view[0]]] += 1
                bPnts0.push_back(pnt_view[0])
                bPnts1.push_back(pnt_view[1])
                pnt_view[0] += 1
            pnt_view[1] += 1

    lk_np = np.zeros(regNum, np.float64)
    regL_np = np.zeros(regNum, np.int64)
    bReg1u = np.zeros((height, width), np.uint8)
    cdef unsigned char [:,::1]bReg1u_view = bReg1u

    lk_np = np.divide(np.multiply(ybNum,yR),regs_vX)
    regL_np = np.where(np.divide(lk_np,thr)>1,255,0)
    bReg1u = np.take(regL_np,regIdx1i)

    with nogil:
        for iii in range(bPnts0.size()):
            bReg1u_view[bPnts1[iii], bPnts0[iii]] = 255
    return regSal1d_np,bReg1u

###基于图的图像分割###
#https://blog.csdn.net/wsp_1138886114/article/details/103546747

cdef packed struct edges:
    int a
    int b
    float w

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef segmentImage(np.ndarray[np.float32_t, ndim=3] smImg3f,float seg_K,int min_size):
    cdef:
        int height = smImg3f.shape[0]
        int width  = smImg3f.shape[1]
        int area = height*width
        int graph_edges_len = 4*area-3*(height+width)+2
        np.ndarray graph_edges = np.ndarray((graph_edges_len,),dtype=[('a', 'i4'),('b', 'i4'),('w', 'f4')])
        edges [:] edges_view = graph_edges
        int num = 0, x = 0, y = 0

    with nogil:
    #Build__graph
        for y in range(height):
            for x in range(width):
                if x < width - 1:
                    edges_view[num].a = y * width + x
                    edges_view[num].b = y * width + (x + 1)
                    edges_view[num].w = sqrt(pow((smImg3f[y,x,0]-smImg3f[y,x+1,0]),2)+ \
                                             pow((smImg3f[y,x,1]-smImg3f[y,x+1,1]),2)+ \
                                             pow((smImg3f[y,x,2]-smImg3f[y,x+1,2]),2))
                    num += 1
                if y < height - 1:
                    edges_view[num].a = y * width + x
                    edges_view[num].b = (y + 1) * width + x
                    edges_view[num].w = sqrt(pow((smImg3f[y,x,0]-smImg3f[y+1,x,0]),2)+ \
                                             pow((smImg3f[y,x,1]-smImg3f[y+1,x,1]),2)+ \
                                             pow((smImg3f[y,x,2]-smImg3f[y+1,x,2]),2))
                    num += 1
                if (x < (width - 1)) and (y < (height - 1)):
                    edges_view[num].a = y * width + x
                    edges_view[num].b = (y + 1) * width + (x + 1)
                    edges_view[num].w = sqrt(pow((smImg3f[y,x,0]-smImg3f[y+1,x+1,0]),2)+ \
                                             pow((smImg3f[y,x,1]-smImg3f[y+1,x+1,1]),2)+ \
                                             pow((smImg3f[y,x,2]-smImg3f[y+1,x+1,2]),2))
                    num += 1
                if (x < (width - 1)) and y > 0:
                    edges_view[num].a = y * width + x
                    edges_view[num].b = (y - 1) * width + (x + 1)
                    edges_view[num].w = sqrt(pow((smImg3f[y,x,0]-smImg3f[y-1,x+1,0]),2)+ \
                                             pow((smImg3f[y,x,1]-smImg3f[y-1,x+1,1]),2)+ \
                                             pow((smImg3f[y,x,2]-smImg3f[y-1,x+1,2]),2))
                    num += 1
    
    #segment_graph
    cdef:
        int i = 0,j = 0,a = 0,b = 0,
    graph_edges = np.sort(np.asarray(edges_view), order='w')
    forest = np.zeros((area, 3), dtype=np.int32)
    forest[:, 1] = np.array(range(area), dtype=np.int32)
    forest[:, 2] = np.ones(area, dtype=np.int32)
    
    cdef edges [:] edges_view2 = graph_edges
    cdef int [:,:] forest_view = forest

    thresholds = np.full(area,seg_K,dtype=np.float32)
    cdef float [:] thresholds_view = thresholds
    
    cdef int comp = 0,idxNum = 0
    cdef int y1=0, x1=0, xi=0,yi=0,i_count=0
    cdef vector[int] marker = []
    cdef unordered_map[int, int] marker_dict
    cdef pair[int, int] pair_marker
    imgIdx_np = np.zeros((smImg3f.shape[0], smImg3f.shape[1]), np.int32)
    cdef int[:,:] imgIdx_view = imgIdx_np
    with nogil:
        for i in range(graph_edges_len):
            while edges_view2[i].a!=forest_view[edges_view2[i].a,1]:
                edges_view2[i].a = forest_view[edges_view2[i].a, 1]
            while edges_view2[i].b!=forest_view[edges_view2[i].b,1]:
                edges_view2[i].b = forest_view[edges_view2[i].b, 1]
            if edges_view2[i].a != edges_view2[i].b:
                if edges_view2[i].w <= thresholds_view[edges_view2[i].a] and edges_view2[i].w <= thresholds_view[edges_view2[i].b]:
                    if (forest_view[edges_view2[i].a, 0] > forest_view[edges_view2[i].b, 0]):
                        forest_view[edges_view2[i].b, 1] = edges_view2[i].a
                        forest_view[edges_view2[i].a, 2] += forest_view[edges_view2[i].b, 2]
                    else:
                        forest_view[edges_view2[i].a, 1] = edges_view2[i].b
                        forest_view[edges_view2[i].b, 2] += forest_view[edges_view2[i].a, 2]
                        if forest_view[edges_view2[i].a, 0] == forest_view[edges_view2[i].b, 0]:
                            forest_view[edges_view2[i].b, 0] += 1
                    while edges_view2[i].a!=forest_view[edges_view2[i].a,1]:
                        edges_view2[i].a = forest_view[edges_view2[i].a, 1]
                    thresholds_view[edges_view2[i].a] = edges_view2[i].w + seg_K/forest_view[edges_view2[i].a,2]
        for j in range(graph_edges_len):
            while edges_view2[j].a!=forest_view[edges_view2[j].a,1]:
                edges_view2[j].a = forest_view[edges_view2[j].a, 1]
            while edges_view2[j].b!=forest_view[edges_view2[j].b,1]:
                edges_view2[j].b = forest_view[edges_view2[j].b, 1]    
            if ((edges_view2[j].a != edges_view2[j].b) and ((forest_view[edges_view2[j].a,2] < min_size) or (forest_view[edges_view2[j].b,2] < min_size))):
                if (forest_view[edges_view2[j].a, 0] > forest_view[edges_view2[j].b, 0]):
                    forest_view[edges_view2[j].b, 1] = edges_view2[j].a
                    forest_view[edges_view2[j].a, 2] += forest_view[edges_view2[j].b, 2]
                else:
                    forest_view[edges_view2[j].a, 1] = edges_view2[j].b
                    forest_view[edges_view2[j].b, 2] += forest_view[edges_view2[j].a, 2]
                    if forest_view[edges_view2[j].a, 0] == forest_view[edges_view2[j].b, 0]:
                        forest_view[edges_view2[j].b, 0] += 1

    #分割块分配随机颜色
        for y1 in range(height):
            for x1 in range(width):
                comp = y1 * width + x1
                while (comp != forest_view[comp, 1]):
                    comp = forest_view[comp, 1]

                for xi in range(marker.size()):
                    if comp == marker[xi]:
                        yi += 1
                if yi == 0:
                    marker.push_back(comp)
                    pair_marker.first = comp
                    pair_marker.second = i_count
                    marker_dict.insert(pair_marker)
                    i_count +=1
                imgIdx_view[y1, x1] = marker_dict[comp]
                yi=0
    idxNum = marker.size()
    return idxNum, imgIdx_np

###Cython 颜色量化（255*255*255 颜色量化转成 12*12*12）###
#https://blog.csdn.net/wsp_1138886114/article/details/103223244

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unordered_map[int,int] color_exchange(int maxNum,int len_num,
                                           int[:,:] color3i,
                                           int [:] num_values):
    cdef:
        unordered_map[int, int] pallet_dict
        pair[int, int] entry
        int i
        int similarIdx = 0
        int similarVal = 0
        int dist_ij = 0
        Py_ssize_t ii = 0
        Py_ssize_t jj = 0
        pair[int, int] entry_exc
    with nogil:
        for i in range(maxNum):
            entry.first = num_values[i]
            entry.second = i
            pallet_dict.insert(entry)
        for ii in range(maxNum,len_num):
            similarIdx = 0
            similarVal = (1 << 31) - 1
            for jj in range(maxNum):
                dist_ij = (color3i[ii][0] - color3i[jj][0])**2 + (color3i[ii][1] - color3i[jj][1])**2 +(color3i[ii][2] - color3i[jj][2])**2
                if dist_ij < similarVal:
                    similarVal = dist_ij
                    similarIdx = jj

            entry_exc.first = num_values[ii]
            entry_exc.second = pallet_dict[num_values[similarIdx]]
            pallet_dict.insert(entry_exc)
        return pallet_dict

@cython.boundscheck(False)
@cython.wraparound(False)
cdef computebin(float[:,:,:] img3f,
                int [:, ::1] idx1i,
                Py_ssize_t height,
                Py_ssize_t width):
    
    bincount_np = np.zeros(1728,dtype=np.int32)
    cdef int[::1]bincount = bincount_np
    cdef vector[int] v_frequency
    cdef vector[int] v_color_value

    cdef Py_ssize_t x, y,i,j
    cdef int non=0
    with nogil:
        for y in prange(height):
            for x in range(width):
                idx1i[y,x] = <int>(img3f[y,x,0]*11.9999)*144 + <int>(img3f[y,x,1]*11.9999) *12 + <int>(img3f[y,x,2]*11.9999)
                bincount[idx1i[y,x]]+=1
        for non in range(1728):
            if bincount[non]!=0:
                v_frequency.push_back(bincount[non])
                v_color_value.push_back(non)
    bincount_npy = np.array([np.asarray(v_frequency),np.asarray(v_color_value)],dtype=np.int32)
    return idx1i,bincount_npy

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int count(int[:]num,maxDropNum):
    cdef int add_accumulate = 0
    cdef int i_count = 0
    while add_accumulate<maxDropNum:
        add_accumulate +=num[i_count]
        i_count +=1
    return i_count

@cython.boundscheck(False)
@cython.wraparound(False)
cdef new_color(int [:,::1] idx1i,
               int maxNum,
               Py_ssize_t height,
               Py_ssize_t width,
               unordered_map[int,int] pallet_dict):
    cdef new_idx_np = np.zeros_like(idx1i)
    cdef colorNum_np = np.zeros((1,maxNum),np.int32)
    
    cdef int [:,::1] new_idx   = new_idx_np
    cdef int [:,::1] colorNum  = colorNum_np
    
    cdef Py_ssize_t x, y
    with nogil:
        for y in range(height):
            for x in range(width):
                new_idx[y,x] = pallet_dict[idx1i[y,x]]
                colorNum[0,new_idx[y,x]] += 1
    return new_idx_np,colorNum_np

cpdef Quantize(np.ndarray[np.float32_t, ndim=3] img3f_np):
    cdef:
        Py_ssize_t height = img3f_np.shape[0]
        Py_ssize_t width  = img3f_np.shape[1]
        int len_num = 0, maxNum = 0, maxDropNum = 0 ,accumulate = 0,arg_sum_pallett=0
        float [:,:,:]img3f = img3f_np

    idx1i_np = np.zeros((height, width),dtype=np.int32)
    cdef int [:,::1] idx1i = idx1i_np


    #统计像素出现频数
    idx1i,bincount_npy = computebin(img3f,idx1i,height,width)
    bincount_npy = bincount_npy[:,bincount_npy[0].argsort()]
    frequency = bincount_npy[0]
    color_value = bincount_npy[1]
    
    len_num = maxNum = len(frequency)                             # 所有颜色值出现频数的排序（从小到大）
    maxDropNum = int(np.round(height * width * 0.05))             # 设置删除最大元素阈值
    cdef int [:] sort_pallet = frequency
    i_count = count(frequency,maxDropNum)
    maxNum = len_num-i_count+1                                    # 后5%的颜色值数量
    num_values = color_value[::-1]                                # 所有高频次颜色值的位置（颜色值降序）

    maxNum = 256 if maxNum > 256 else maxNum
    if maxNum <= 10:
        maxNum = 10 if len_num > 10 else len_num

    #计算像素距离
    color3i_init0 = np.zeros(len_num,dtype=np.int32)
    color3i_init1 = np.zeros(len_num,dtype=np.int32)
    color3i_init2 = np.zeros(len_num,dtype=np.int32)
    color3i_np = np.zeros((len_num,3),dtype=np.int32)

    color3i_init0 = np.divide(num_values,144).astype(np.int32)
    color3i_init1 = np.divide(np.mod(num_values,144),12).astype(np.int32)
    color3i_init2 = np.mod(num_values , 12).astype(np.int32)
    color3i = np.array([color3i_init0,color3i_init1,color3i_init2]).T

    pallet_dict = color_exchange(maxNum,len_num,color3i,num_values)
    color3f = np.zeros((1,maxNum,3),np.float32)
    
    new_idx_np,colorNum_np = new_color(idx1i,maxNum,height,width,pallet_dict)
    np.add.at(color3f[0], new_idx_np, img3f)
    color3f[0] /= (colorNum_np.T).astype(np.float32)
    return color3f.shape[1],new_idx_np,color3f,colorNum_np

###区域平滑###

cpdef Smooth(np.ndarray[np.float32_t, ndim=3]img3f,
             np.ndarray[np.float64_t, ndim=2]sal1f,
             int[:,:]regIdx1i, 
             int regNum, 
             float delta):
    cdef:
        int binN = 0
        int tmpNum = 0
        int n = 0
        int height = img3f.shape[0]
        int width = img3f.shape[1]
        np.ndarray[np.int32_t,   ndim=2] idx1i
        np.ndarray[np.float32_t, ndim=3] binColor3f
        np.ndarray[np.int32_t,   ndim=2] colorNums1i

    binN, idx1i, binColor3f, colorNums1i = Quantize(img3f)         # 区域颜色量化
    _colorSal = np.zeros((1, binN),dtype = np.float64)

    np.add.at(_colorSal[0],idx1i,sal1f)
    _colorSal[0] = np.divide(_colorSal[0],colorNums1i[0])
    cv2.normalize(_colorSal, _colorSal, 0, 1, cv2.NORM_MINMAX)     # 区域颜色归一化
    
    cv2.cvtColor(binColor3f, cv2.COLOR_BGR2Lab, binColor3f)        # BGR2Lab计算区域颜色之间的距离
    color3fv_reshape = np.zeros((binN, 3),dtype = np.float32)
    color3fv_reshape = binColor3f.reshape(binN, 3)[:binN]
    
    similar_dist = np.zeros((binN, binN),dtype = np.float64)
    similar_dist_sort = np.zeros((binN, binN),dtype = np.float64)
    similar_dist_argsort = np.zeros((binN, binN),dtype = np.int64)
    
    similar_dist = squareform(pdist(color3fv_reshape))
    similar_dist_sort = np.sort(similar_dist)                      # 排序
    similar_dist_argsort = np.argsort(similar_dist)                # 排序前索引
    cv2.cvtColor(binColor3f, cv2.COLOR_Lab2BGR, binColor3f)        # Lab2BGR颜色转回GBR

    #SmoothBySaliency
    
    if binN < 2:
        return
    tmpNum = int(binN * delta+0.5)                                 # tmpNum 占比0.25的变化的颜色值数量
    n = tmpNum if tmpNum > 2 else 2

    #获取相似度距离排序后的索引
    idx = np.zeros((binN, n),dtype = np.int64)
    val_n = np.zeros((binN, n),dtype = np.float64)
    w_n = np.zeros((binN, n),dtype = np.int32)
    
    idx = similar_dist_argsort[:, :n]
    val_n = np.take(_colorSal, idx)                                # 获取占比前0.25的颜色
    w_n = np.take(colorNums1i[0], idx)

    similar_nVal = np.zeros((binN, n),dtype = np.float64)
    every_Dist = np.zeros((binN, n),dtype = np.float64)
    totalDist = np.zeros((binN,),dtype = np.float64)
    totalWeight = np.zeros((binN,),dtype = np.int32)
    valCrnt = np.zeros((binN,),dtype = np.float64)
    newSal_img = np.zeros((1, binN), np.float64)
    
    similar_nVal = similar_dist_sort[:, :n]
    totalDist = np.sum(similar_nVal, axis=1)                       # totalDist 距离计算排序后的每行求和
    totalWeight = np.sum(w_n, axis=1)                              # totalWeight 距离计算排序后的每行求和
    every_Dist = (np.tile(totalDist[:, np.newaxis], (1, n)) - similar_nVal)*w_n
    valCrnt = np.sum(val_n[:, :n] * every_Dist, axis=1)
    newSal_img[0] = valCrnt / (totalDist * totalWeight)

    cv2.normalize(newSal_img, _colorSal, 0, 1, cv2.NORM_MINMAX)    # 归一化
    sal1f = np.take(_colorSal[0], idx1i)

    #区域平滑
    saliecy = np.zeros(regNum,dtype=np.float64)
    counter = np.zeros(regNum,dtype=np.int32)
    np.add.at(saliecy, regIdx1i, sal1f)
    np.add.at(counter, regIdx1i, 1)
    saliecy = np.divide(saliecy,counter)
    cv2.normalize(saliecy, saliecy, 0, 1, cv2.NORM_MINMAX)
    sal1f = np.take(saliecy,regIdx1i)
    return sal1f
