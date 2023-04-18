#include <iostream>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

struct fragCount
{
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		float x1, y1, x2, y2, x3, y3;
		x1 = thrust::get<0>(thrust::get<0>(t));
		y1 = thrust::get<1>(thrust::get<0>(t));
		x2 = thrust::get<0>(thrust::get<1>(t));
		y2 = thrust::get<1>(thrust::get<1>(t));
		x3 = thrust::get<0>(thrust::get<2>(t));
		y3 = thrust::get<1>(thrust::get<2>(t));
		float minY = y1 < y2 ? y1 : (y2 < y3 ? y2 : y3);
		float maxY = y1 > y2 ? y1 : (y2 > y3 ? y2 : y3);
		int low = ceil(minY);
		int high = floor(maxY);
		int frags = 0;
		for(int i = low; i <= high; i++)
		{
			float ed1, ed2, ed3;
			bool e1 = false, e2 = false, e3 = false;
			if(y1 < y2 && i >= y1 && i <= y2 || y1 > y2 && i >= y2 && i <= y1) 
			{
				ed1 = (i - y1) * (x2-x1) / (y2-y1) + x1;
				e1 = true;
			}
			if(y2 < y3 && i >= y2 && i <= y3 || y2 > y3 && i >= y3 && i <= y2) 
			{
				ed2 = (i - y2) * (x3-x2) / (y3-y2) + x2;
				e2 = true;
			}
			if(y1 < y3 && i >= y1 && i <= y3 || y1 > y3 && i >= y3 && i <= y1) 
			{
				ed2 = (i - y1) * (x3-x1) / (y3-y1) + x1;
				ed3 = true;
			}

			float end1, end2;
			
			if(e1 && e2 && e3)
			{
				float eq = ed1 == ed2 ? ed1 : ed3;
				float neq = ed1 == ed2 ? ed3 : ed1;
				end1 = eq < neq ? eq : neq;
				end2 = eq < neq ? neq : eq;
			}
			else if (e1 && e2)
			{
				end1 = ed1 < ed2 ? ed1 : ed2;
				end2 = ed1 < ed2 ? ed2 : ed1;
			}
			else if(e2 && e3)
			{
				end1 = ed2 < ed3 ? ed2 : ed3;
				end2 = ed2 < ed3 ? ed3 : ed2;
			}
			else if(e1 && e3)
			{
				end1 = ed1 < ed3 ? ed1 : ed3;
				end2 = ed1 < ed3 ? ed3 : ed1;
			}
			
			frags += floor(end2) - ceil(end1);
		}

		thrust::get<3>(t) = frags;
	}
};

struct rasterize 
{
	thrust::device_vector<thrust::tuple<int,int,float,char,char,char>> *frag;
	__host__ __device__
	rasterize(thrust::device_vector<thrust::tuple<int,int,float,char,char,char>> *_frag) : frag(_frag) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		float x1, y1, z1, x2, y2, z2, x3, y3, z3;
		x1 = thrust::get<0>(thrust::get<0>(t));
		y1 = thrust::get<1>(thrust::get<0>(t));
		z1 = thrust::get<2>(thrust::get<0>(t));
		x2 = thrust::get<0>(thrust::get<1>(t));
		y2 = thrust::get<1>(thrust::get<1>(t));
		z2 = thrust::get<2>(thrust::get<1>(t));
		x3 = thrust::get<0>(thrust::get<2>(t));
		y3 = thrust::get<1>(thrust::get<2>(t));
		z3 = thrust::get<2>(thrust::get<2>(t));
		//calculate triangle plane
		float x_coe = ((y2-y1)*(z3-z1)-(y3-y1)*(z2-z1));
		float y_coe = ((x2-x1)*(z3-z1)-(x3-x1)*(z2-z1));
		float z_coe = ((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1));
		float minY = y1 < y2 ? y1 : (y2 < y3 ? y2 : y3);
		float maxY = y1 > y2 ? y1 : (y2 > y3 ? y2 : y3);
		int low = ceil(minY);
		int high = floor(maxY);
		int count = thrust::get<3>(t);
		char r = thrust::get<0>(thrust::get<4>(t));
		char g = thrust::get<1>(thrust::get<4>(t));
		char b = thrust::get<2>(thrust::get<4>(t));
		for(int i = low; i <= high; i++)
		{
			float ed1, ed2, ed3;
			bool e1 = false, e2 = false, e3 = false;
			if(y1 < y2 && i >= y1 && i <= y2 || y1 > y2 && i >= y2 && i <= y1) 
			{
				ed1 = (i - y1) * (x2-x1) / (y2-y1) + x1;
				e1 = true;
			}
			if(y2 < y3 && i >= y2 && i <= y3 || y2 > y3 && i >= y3 && i <= y2) 
			{
				ed2 = (i - y2) * (x3-x2) / (y3-y2) + x2;
				e2 = true;
			}
			if(y1 < y3 && i >= y1 && i <= y3 || y1 > y3 && i >= y3 && i <= y1) 
			{
				ed2 = (i - y1) * (x3-x1) / (y3-y1) + x1;
				ed3 = true;
			}

			float end1, end2;
			
			if(e1 && e2 && e3)
			{
				float eq = ed1 == ed2 ? ed1 : ed3;
				float neq = ed1 == ed2 ? ed3 : ed1;
				end1 = eq < neq ? eq : neq;
				end2 = eq < neq ? neq : eq;
			}
			else if (e1 && e2)
			{
				end1 = ed1 < ed2 ? ed1 : ed2;
				end2 = ed1 < ed2 ? ed2 : ed1;
			}
			else if(e2 && e3)
			{
				end1 = ed2 < ed3 ? ed2 : ed3;
				end2 = ed2 < ed3 ? ed3 : ed2;
			}
			else if(e1 && e3)
			{
				end1 = ed1 < ed3 ? ed1 : ed3;
				end2 = ed1 < ed3 ? ed3 : ed1;
			}

			int end = floor(end2), start = ceil(end1);
			for(int j = start; j <= end; j++)
			{
				float k = z1 - (x_coe*(j-x1)+y_coe*(i-y1))/z_coe;
				(*frag)[count++] = thrust::make_tuple(j, i, k, r, g, b);
			}
		}
	}
};

struct rowCount
{
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		float y1, y2, y3;
		y1 = thrust::get<1>(thrust::get<0>(t));
		y2 = thrust::get<1>(thrust::get<1>(t));
		y3 = thrust::get<1>(thrust::get<2>(t));
		float minY = y1 < y2 ? y1 : (y2 < y3 ? y2 : y3);
		float maxY = y1 > y2 ? y1 : (y2 > y3 ? y2 : y3);
		thrust::get<3>(t) = floor(maxY) - ceil(minY);
	}
};

struct colCount
{
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		float x1, y1, x2, y2, x3, y3;
		x1 = thrust::get<0>(thrust::get<0>(t));
		y1 = thrust::get<1>(thrust::get<0>(t));
		x2 = thrust::get<0>(thrust::get<1>(t));
		y2 = thrust::get<1>(thrust::get<1>(t));
		x3 = thrust::get<0>(thrust::get<2>(t));
		y3 = thrust::get<1>(thrust::get<2>(t));
		int row = thrust::get<3>(t);
		float ed1, ed2, ed3;
		bool e1 = false, e2 = false, e3 = false;
		if(y1 < y2 && row >= y1 && row <= y2 || y1 > y2 && row >= y2 && row <= y1) 
		{
			ed1 = (row - y1) * (x2-x1) / (y2-y1) + x1;
			e1 = true;
		}
		if(y2 < y3 && row >= y2 && row <= y3 || y2 > y3 && row >= y3 && row <= y2) 
		{
			ed2 = (row - y2) * (x3-x2) / (y3-y2) + x2;
			e2 = true;
		}
		if(y1 < y3 && row >= y1 && row <= y3 || y1 > y3 && row >= y3 && row <= y1) 
		{
			ed2 = (row - y1) * (x3-x1) / (y3-y1) + x1;
			ed3 = true;
		}

		float end1, end2;
		
		if(e1 && e2 && e3)
		{
			float eq = ed1 == ed2 ? ed1 : ed3;
			float neq = ed1 == ed2 ? ed3 : ed1;
			end1 = eq < neq ? eq : neq;
			end2 = eq < neq ? neq : eq;
		}
		else if (e1 && e2)
		{
			end1 = ed1 < ed2 ? ed1 : ed2;
			end2 = ed1 < ed2 ? ed2 : ed1;
		}
		else if(e2 && e3)
		{
			end1 = ed2 < ed3 ? ed2 : ed3;
			end2 = ed2 < ed3 ? ed3 : ed2;
		}
		else if(e1 && e3)
		{
			end1 = ed1 < ed3 ? ed1 : ed3;
			end2 = ed1 < ed3 ? ed3 : ed1;
		}
		
		thrust::get<4>(t) = floor(end2) - ceil(end1);
	}
};

void expand_int
	(thrust::device_vector<int>::iterator map,
	 thrust::device_vector<int>::iterator pred,
	 thrust::device_vector<int>::iterator start,
	 thrust::device_vector<int>::iterator end,
	 int num)
{
	thrust::scatter_if
		(thrust::counting_iterator<int>(0),
		 thrust::counting_iterator<int>(num),
		 map,
		 pred,
		 start);

	thrust::inclusive_scan
		(start,
		 end,
		 start,
		 thrust::maximum<int>());
}	

void index_int
	(thrust::device_vector<int>::iterator map,
	 thrust::device_vector<int>::iterator src,
	 thrust::device_vector<int>::iterator out,
	 int num)
{
	thrust::transform
		(thrust::counting_iterator<int>(0),
		 thrust::counting_iterator<int>(num),
		 thrust::make_permutation_iterator(src, map),
		 out,
		 thrust::minus<int>());
}


void print_int_vec(thrust::device_vector<int>::iterator start,
		   thrust::device_vector<int>::iterator end)
{
	for(; start < end; start++)
		std::cout << *start << " ";
	std::cout << std::endl;
}

int main()
{
	thrust::device_vector<thrust::tuple<float, float, float>> p1(2);
	thrust::device_vector<thrust::tuple<float, float, float>> p2(2);
	thrust::device_vector<thrust::tuple<float, float, float>> p3(2);
	thrust::device_vector<thrust::tuple<char, char, char>> color(2);

	p1[0] = thrust::make_tuple(0,0,0);
	p2[0] = thrust::make_tuple(5,5,0);
	p3[0] = thrust::make_tuple(10,0,0);
	p1[1] = thrust::make_tuple(0,0,0);
	p2[1] = thrust::make_tuple(5,5,0);
	p3[1] = thrust::make_tuple(10,0,0);

	color[0] = thrust::make_tuple(255,0,0);
	color[1] = thrust::make_tuple(0,0,255);

	thrust::device_vector<int> frags(2);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(p1.begin(), p2.begin(), p3.begin(), frags.begin())),
			 thrust::make_zip_iterator(thrust::make_tuple(p1.end(), p2.end(), p3.end(), frags.end())),
			 fragCount());
	std::cout << "# frags by triange: " << frags[0] << ", " << frags[1] << std::endl;

	thrust::device_vector<int> write_index(2);

	thrust::exclusive_scan(frags.begin(), frags.end(), write_index.begin());

	std::cout << "write position by triange: " << write_index[0] << ", " << write_index[1] << std::endl;

	int fragments = write_index[1] + frags[1];
	
	std::cout << "Number of fragments: " << fragments << std::endl;
	/*

	thrust::device_vector<int> frag_pos(fragments);

	thrust::scatter_if
		(thrust::counting_iterator<int>(0),
		 thrust::counting_iterator<int>(2),
		 write_index.begin(),
		 frags.begin(),
		 frag_pos.begin());

	thrust::inclusive_scan
		(frag_pos.begin(),
		 frag_pos.end(),
		 frag_pos.begin(),
		 thrust::maximum<int>());
	
	thrust::device_vector<int> frag_ind(fragments);

	thrust::transform
		(thrust::counting_iterator<int>(0),
		 thrust::counting_iterator<int>(fragments),
		 thrust::make_permutation_iterator(write_index.begin(),
			 			   frag_pos.begin()),
		 frag_ind.begin(),
		 thrust::minus<int>());
		 */
	
	thrust::device_vector<int> rows(2);
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(p1.begin(), p2.begin(), p3.begin(), rows.begin())),
			 thrust::make_zip_iterator(thrust::make_tuple(p1.end(), p2.end(), p3.end(), rows.end())),
			 rowCount());
	
	for(int i = 0; i < 2; i++)
		std::cout << rows[i] << " ";
	std::cout << std::endl;

	thrust::device_vector<int> row_off(2);
	thrust::exclusive_scan(rows.begin(), rows.end(), row_off.begin());
	for(int i = 0; i < 2; i++)
		std::cout << row_off[i] << " ";
	std::cout << std::endl;

	int num_rows = row_off[1] + rows[1];

	thrust::device_vector<int> tri_ptr(num_rows);
	
	expand_int(row_off.begin(), rows.begin(), tri_ptr.begin(), tri_ptr.end(), 2);

	for(int i = 0; i < num_rows; i++)
		std::cout << tri_ptr[i] << " ";
	std::cout << std::endl;	

	thrust::device_vector<int> row_ptr(num_rows);

	index_int(tri_ptr.begin(), row_off.begin(), row_ptr.begin(), num_rows);

	for(int i = 0; i < num_rows; i++)
		std::cout << row_ptr[i] << " ";
	std::cout << std::endl;	

	thrust::device_vector<int> col_count(num_rows);

	thrust::for_each
		(thrust::make_zip_iterator
		 	(thrust::make_permutation_iterator(p1.begin(), tri_ptr.begin()),
			 thrust::make_permutation_iterator(p2.begin(), tri_ptr.begin()),
			 thrust::make_permutation_iterator(p3.begin(), tri_ptr.begin()),
			 row_ptr.begin(),
			 col_count.begin()),
		thrust::make_zip_iterator
		 	(thrust::make_permutation_iterator(p1.begin(), tri_ptr.end()),
			 thrust::make_permutation_iterator(p2.begin(), tri_ptr.end()),
			 thrust::make_permutation_iterator(p3.begin(), tri_ptr.end()),
			 row_ptr.end(),
			 col_count.end()),
		colCount());

	print_int_vec(col_count.begin(), col_count.end());

}
