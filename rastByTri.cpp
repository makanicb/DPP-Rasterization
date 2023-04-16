#include <iostream>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/for_each.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>

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
	thrust::device_vector<thrust::tuple<int,int,char,char,char>> *frag;
	__host__ __device__
	rasterize(thrust::device_vector<thrust::tuple<int,int,char,char,char>> *_frag) : frag(_frag) {}

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
				(*frag)[count++] = thrust::make_tuple(j, i, r, g, b);
		}
	}
};

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

	thrust::device_vector<thrust::tuple<int,int,char,char,char>> frag_pos(fragments);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(p1.begin(),
				       			     	      p2.begin(),
								      p3.begin(),
								      write_index.begin(),
								      color.begin())),
			 thrust::make_zip_iterator(thrust::make_tuple(p1.end(),
					 			      p2.end(),
								      p3.end(),
								      write_index.end(),
								      color.end())),
			 rasterize(&frag_pos));

	for(int i = 0; i < fragments; i++)
	{
		thrust::tuple<int,int,char,char,char> test = frag_pos[i];
		std::cout<<i+1<<":"<<thrust::get<0>(test)<<","<<thrust::get<1>(test)<<std::endl;
		std::cout<<i+1<<":"<<(int)thrust::get<2>(test)<<","<<
			(int)thrust::get<3>(test)<<","<<(int)thrust::get<4>(test)<<std::endl;
	}
}
