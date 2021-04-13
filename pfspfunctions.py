import numpy as np

#################### METRICS CALCULATION FUNCTION ####################
######################################################################

def calculateMetrics(matrixo,weights,cum_mat=[],begin=0):
    if begin<=0:
        cum_mat = np.copy(matrixo)
        for i in range(np.size(matrixo,0)):
            for j in range(np.size(matrixo,1)):
                if i == 0:
                    if j == 0:
                        continue
                    cum_mat[i,j] = cum_mat[i,j-1] + cum_mat[i,j]
                else:
                    if j == 0:
                        cum_mat[i,j] = cum_mat[i-1,j] + cum_mat[i,j]
                    else:
                        cum_mat[i,j] = max(cum_mat[i-1,j],cum_mat[i,j-1]) + cum_mat[i,j]
    else:
        for i in range(begin,np.size(cum_mat,0)):
            for j in range(np.size(cum_mat,1)):
                if j == 0:
                    cum_mat[i,j] = cum_mat[i-1,j] + matrixo[i,j]
                else:
                    cum_mat[i,j] = max(cum_mat[i-1,j],cum_mat[i,j-1]) + matrixo[i,j]
                
    makespan = int(cum_mat[-1,-1])
    sum_compl = sum(np.multiply(cum_mat[:,-1],weights[:,-1]))
    return makespan, sum_compl, cum_mat
        
    

#################### NEIGHBOURHOOD FUNCTIONS ####################
#################################################################

def exchange(matrixo,i,j):
    """Exchanges the rows i and j from the matrix 'matrixo'. Naturally,
    i and j are integers between 0 and size(matrix,0)."""
    
    matrix = np.copy(matrixo)
    buf = np.copy(matrix[i])
    matrix[i] = matrix[j]
    matrix[j] = buf
    return matrix

def transpose(matrixo,i,lr):
    """Transposes the rows (of matrixo) i and i-1 if lr is set to -1, and i and i+1 
    if lr is set to 1."""
    
    matrix = np.copy(matrixo)
    buf = np.copy(matrix[i])
    if lr == 1:
        j=i+1
    else:
        j=i-1
    matrix[i] = matrix[j]
    matrix[j] = buf
    return matrix

def insert(matrixo,i,abs_loc):
    """Inserts row i (of matrixo) just before row abs_loc."""
    
    matrix = np.copy(matrixo)
    buf = np.copy(matrix[i])
    matrix = np.delete(matrix,i,0)
    matrix = np.insert(matrix,abs_loc,buf,0)
    return matrix



#################### PIVOTING RULE RELATED FUNCTIONS ####################
#########################################################################

def calculateFirstNeighbour(matrixo, weightso, nmethod):
    # nmethod is the neighbour method, 'transpose', 'insert' or 'exchange'
    nJobs = np.size(matrixo,0)
    matrix = np.copy(matrixo)
    weights = np.copy(weightso)
    _,summ,cum_mat = calculateMetrics(matrix,weights) 
    
    if nmethod == 'insert':
        for i in range(nJobs):
            for j in range(nJobs):
                if i == j:
                    continue
                else:
                    mat2 = insert(matrix,i,j)
                    weights2 = insert(weights,i,j)
                    cum_mat2 = insert(cum_mat,i,j)
                    _,nextsum,next_cummat = calculateMetrics(mat2,weights2,cum_mat2,min(i,j))
                if nextsum < summ:
                    matrix=mat2
                    weights=weights2
                    cum_mat=next_cummat
                    summ = nextsum
                    return matrix,weights,summ
        return matrix,weights,summ
    
    elif nmethod == 'exchange':
        for i in range(nJobs):
            for j in range(i,nJobs):
                if i == j:
                    continue
                else:
                    mat2 = exchange(matrix,i,j)
                    weights2 = exchange(weights,i,j)
                    cum_mat2 = exchange(cum_mat,i,j)
                    _,nextsum,next_cummat = calculateMetrics(mat2,weights2,cum_mat2,min(i,j))
                if nextsum < summ:
                    matrix=mat2
                    weights=weights2
                    cum_mat=next_cummat
                    summ = nextsum
                    return matrix,weights,summ
        return matrix,weights,summ
    
    elif nmethod == 'transpose':
        for i in range(nJobs):
            for j in [-1, 1]:
                if i == 0 and j == -1:
                    continue
                elif i == nJobs-1 and j == 1:
                    continue
                else:
                    mat2 = transpose(matrix,i,j)
                    weights2 = transpose(weights,i,j)
                    cum_mat2 = transpose(cum_mat,i,j)
                    _,nextsum,next_cummat = calculateMetrics(mat2,weights2,cum_mat2,min(i,i+j))
                if nextsum < summ:
                    matrix=mat2
                    weights=weights2
                    cum_mat=next_cummat
                    summ = nextsum
                    return matrix,weights,summ     
        return matrix,weights,summ
                
    else:
        print('Neighbourhood method not valid. Please choose between ''insert'', ''exchange'' or ''transpose''.')
        return -1
                
def calculateBestNeighbour(matrixo, weightso, nmethod):
    # nmethod is the neighbour method, transpose, insert or exchange
    nJobs = np.size(matrixo,0)
    matrix = np.copy(matrixo)
    weights = np.copy(weightso)
    _,best_sum,cum_mat = calculateMetrics(matrix,weights) 
    sumo = np.copy(best_sum)
    
    if nmethod == 'insert':
        for i in range(nJobs):
            for j in range(nJobs):
                if i == j:
                    continue                    
                else:
                    mat2 = insert(matrix,i,j)
                    weights2 = insert(weights,i,j)
                    cum_mat2 = insert(cum_mat,i,j)
                    _,nextsum,next_cummat = calculateMetrics(mat2,weights2,cum_mat2,min(i,j))
                if nextsum < best_sum:
                    best_matrix=mat2
                    best_weights=weights2
                    best_cum_mat=next_cummat
                    best_sum = nextsum
        if best_sum == sumo:
            best_matrix=matrix
            best_weights=weights
            
        return best_matrix,best_weights,best_sum
    
    elif nmethod == 'exchange':
        for i in range(nJobs):
            for j in range(i,nJobs):
                if i == j:
                    continue
                else:
                    mat2 = exchange(matrix,i,j)
                    weights2 = exchange(weights,i,j)
                    cum_mat2 = exchange(cum_mat,i,j)
                    _,nextsum,next_cummat = calculateMetrics(mat2,weights2,cum_mat2,min(i,j))
                if nextsum < best_sum:
                    best_matrix=mat2
                    best_weights=weights2
                    best_cum_mat=next_cummat
                    best_sum = nextsum
        if best_sum == sumo:
            best_matrix=matrix
            best_weights=weights
            
        return best_matrix,best_weights,best_sum
    
    elif nmethod == 'transpose':
        for i in range(nJobs):
            for j in [-1,1]:
                if i == 0 and j == -1:
                    continue
                elif i == nJobs-1 and j == 1:
                    continue
                else:
                    mat2 = transpose(matrix,i,j)
                    weights2 = transpose(weights,i,j)
                    cum_mat2 = exchange(cum_mat,i,j)
                    _,nextsum,next_cummat = calculateMetrics(mat2,weights2,cum_mat2,min(i,j))
                if nextsum < best_sum:
                    best_matrix=mat2
                    best_weights=weights2
                    best_cum_mat=next_cummat
                    best_sum = nextsum  
                    
        if best_sum == sumo:
            best_matrix=matrix
            best_weights=weights
            
        return best_matrix,best_weights,best_sum
    
    else:
        print('Neighbourhood method not valid. Please choose between ''insert'', ''exchange'' or ''transpose''.')
        return -1


#################### INITIAL SOLUTION FUNCTION ####################
###################################################################

def initialSolution(instance_path,method):

    fh = instance_path.replace('/','_').split('_')
    if '50' in fh:
        nJobs = 50
    elif '100' in fh:
        nJobs = 100
    else:
        print('Error: could not determine number of jobs automatically.')
            
    instances = open(instance_path,'r')
    instances.readline()

    mat = np.zeros( (nJobs,20) )
    for i in range(nJobs):
        buf = instances.readline().split(' ')[1:-1:2]
        for j in range(len(buf)):
            mat[i][j] = int(buf[j])

    instances.readline()
    weights = np.zeros( (nJobs,2) )
    for i in range(nJobs):
        hey = int(instances.readline().split(' ')[-1])
        weights[i][0], weights[i][1] = i+1, hey
    
    # NOW GENERATING INITIAL SOLUTION
    matex = np.hstack((mat,weights))
    if method == 'srz':
        ind=np.argsort(matex[:,-1])
        matex=matex[ind]
        sumz = []
        mat_sol = matex[0]
        summ = sum(mat_sol[:-2])*mat_sol[-1]
        sumz = np.append(sumz,summ)
        for i in range(1,nJobs):
            if i == 1:
                mat2 = np.vstack((mat_sol,matex[1]))
                summ2 = summ + (mat2[i-1,0] + sum(mat2[i,:-2]) ) * mat2[i,-1]
                mat3 = np.vstack((matex[1],mat_sol))
                summ3 = summ + (mat3[i-1,0] + sum(mat3[i,:-2]) ) * mat3[i,-1]
                if summ2 < summ3:
                    summ = summ2
                    mat_sol = mat2
                    sumz = np.append(sumz,summ2)
                else:
                    summ = summ3
                    mat_sol = mat3
                    sumz = np.append(sumz,summ3)
            else:
                for j in range(i+1):
                    mat2 = np.insert(mat_sol,j,matex[i],0)
                    mat_2,w_2 = np.hsplit(mat2,[20])
                    _,cumsum,_ = calculateMetrics(mat_2,w_2)
                    if j == 0:
                        best_cumsum = cumsum
                        best_mat = mat2
                    if cumsum < best_cumsum:
                        best_cumsum = cumsum
                        best_mat = mat2
                mat_sol = best_mat
                cumsum = best_cumsum
                    
        mat, weights = np.hsplit(mat_sol,[20])
        return mat, weights, cumsum
        
    elif method == 'ri':
        np.random.seed(171)
        np.take(matex,np.random.permutation(matex.shape[0]),axis=0,out=matex)
        mat, weights = np.hsplit(matex,[20])
        _,cumsum,_ = calculateMetrics(mat,weights)
        return mat, weights, cumsum
    
    else:
        print('Error: initializing method not valid. Please use ''--srz'' or ''--ri''.')
        return -1