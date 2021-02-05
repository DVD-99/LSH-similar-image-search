# Similar image search using Locality Sensitive Hashing

A dataset of images ```patches.csv```is avaiable [here](https://drive.google.com/file/d/1ThX7oGWYAH8dXc20SoYHUU0CMx8otLvB/view?usp=sharing)
Each row in this dataset is a 20 × 20 image patch represented as a 400-dimensional vector.

**query_index** is image patches of a particular row.
The average search time for LSH search is less than a linear search for the following query_index of 100, 200, 300…, 1000. I have used the same hash functions for query_index, which computed while calling the hash_data() function. Implemented the lsh_search() function using the PySpark and implemented the linear_search() function using the NumPy array. The linear_search() function runtime is O(n) and the lsh_search() function runtime is O(log n)

The error value as a function of L (for L = 10, 12, 14, ..., 20, with k = 24)
![L Graph](https://github.com/DVD-99/LSH-similar-image-search/blob/main/Lerror.PNG)

The error value as a function of k (for k = 16, 18, 20, 22, 24 with L = 10)
![K Graph](https://github.com/DVD-99/LSH-similar-image-search/blob/main/Kerror.PNG)

Based on the graphs choosing L = 12, K = 22 will give us better results.
