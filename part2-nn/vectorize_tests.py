import time 
import numpy 
import array 
  
# 8 bytes size int 
a = array.array('q') 
for i in range(100000): 
    a.append(i); 
  
b = array.array('q') 
for i in range(100000, 200000): 
    b.append(i) 

# classic dot product of vectors implementation  
tic = time.process_time() 
dot = 0.0; 
  
for i in range(len(a)): 
      dot += a[i] * b[i] 
  
toc = time.process_time() 
  
print("dot_product = "+ str(dot)); 
print("Computation time = " + str(1000*(toc - tic )) + "ms") 

n_tic = time.process_time() 
n_dot_product = numpy.dot(a, b) 
n_toc = time.process_time() 
  
print("\nn_dot_product = "+str(n_dot_product)) 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 

# Outer product 
import time 
import numpy 
import array 
  
a = array.array('i') 
for i in range(200): 
    a.append(i); 
  
b = array.array('i') 
for i in range(200, 400): 
    b.append(i) 
  
# classic outer product of vectors implementation  
tic = time.process_time() 
outer_product = numpy.zeros((200, 200)) 
  
for i in range(len(a)): 
   for j in range(len(b)): 
      outer_product[i][j]= a[i]*b[j] 
  
toc = time.process_time() 
  
print("outer_product = "+ str(outer_product)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms") 
   
n_tic = time.process_time() 
outer_product = numpy.outer(a, b) 
n_toc = time.process_time() 
  
print("outer_product = "+str(outer_product)); 
print("\nComputation time = "+str(1000*(n_toc - n_tic ))+"ms") 

# Element-wise multiplication 
import time 
import numpy 
import array 
  
a = array.array('i') 
for i in range(50000): 
    a.append(i); 
  
b = array.array('i') 
for i in range(50000, 100000): 
    b.append(i) 
  
# classic element wise product of vectors implementation  
vector = numpy.zeros((50000)) 
  
tic = time.process_time() 
  
for i in range(len(a)): 
      vector[i]= a[i]*b[i] 
  
toc = time.process_time() 
  
print("Element wise Product = "+ str(vector)); 
print("\nComputation time = "+str(1000*(toc - tic ))+"ms") 
   
  
n_tic = time.process_time() 
vector = numpy.multiply(a, b) 
n_toc = time.process_time() 
  
print("Element wise Product = "+str(vector)); 
print("\nComputation time = "+str(1000*(n_toc - n_tic ))+"ms") 