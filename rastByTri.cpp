#include <iostream>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#define width 300
#define height 300

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
		int y = ceil(minY) + thrust::get<3>(t);
		float ed1, ed2, ed3;
		bool e1 = false, e2 = false, e3 = false;
		if(y1 < y2 && y >= y1 && y <= y2 || y1 > y2 && y >= y2 && y <= y1) 
		{
			ed1 = (y - y1) * (x2-x1) / (y2-y1) + x1;
			e1 = true;
		}
		if(y2 < y3 && y >= y2 && y <= y3 || y2 > y3 && y >= y3 && y <= y2) 
		{
			ed2 = (y - y2) * (x3-x2) / (y3-y2) + x2;
			e2 = true;
		}
		if(y1 < y3 && y >= y1 && y <= y3 || y1 > y3 && y >= y3 && y <= y1) 
		{
			ed2 = (y - y1) * (x3-x1) / (y3-y1) + x1;
			ed3 = true;
		}

		float end1;
		
		if(e1 && e2 && e3)
		{
			float eq = ed1 == ed2 ? ed1 : ed3;
			float neq = ed1 == ed2 ? ed3 : ed1;
			end1 = eq < neq ? eq : neq;
		}
		else if (e1 && e2)
		{
			end1 = ed1 < ed2 ? ed1 : ed2;
		}
		else if(e2 && e3)
		{
			end1 = ed2 < ed3 ? ed2 : ed3;
		}
		else if(e1 && e3)
		{
			end1 = ed1 < ed3 ? ed1 : ed3;
		}

		int x = ceil(end1) + thrust::get<4>(t);
		float z = z1 - (x_coe*(x-x1)+y_coe*(y1-y))/z_coe;
		thrust::get<5>(t)  = thrust::pair(x, y);
		thrust::get<6>(t) = z;
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

void print_pair_vec(thrust::device_vector<thrust::pair<int,int>>::iterator start,
		    thrust::device_vector<thrust::pair<int,int>>::iterator end)
{
	for(; start < end; start++)
	{
		thrust::pair<int,int> temp = *start;
		std::cout << temp.first << "," << temp.second << "\t";
	}
	std::cout << std::endl;
}

void print_float_vec(thrust::device_vector<float>::iterator start,
		     thrust::device_vector<float>::iterator end)
{
	for(; start < end; start++)
		std::cout << *start << " ";
	std::cout << std::endl;
}
/*
struct key_equality
{
	__host__ __device__	
	bool operator()
		(thrust::pair<thrust::pair<int,int>, int> p1, thrust::pair<thrust::pair<int,int>, int> p2)
	{
		return thrust::get<0>(thrust::get<0>(p1)) == thrust::get<0>(thrust::get<0>(p2)) &&
		       thrust::get<1>(thrust::get<0>(p1)) == thrust::get<1>(thrust::get<0>(p2));
	}
};
*/

struct findPositions
{
	thrust::device_vector<thrust::pair<int,int>>::iterator start;
	thrust::device_vector<thrust::pair<int,int>>::iterator stop;
	findPositions
		(thrust::device_vector<thrust::pair<int,int>>::iterator _start, thrust::device_vector<thrust::pair<int,int>>::iterator _stop)
		: start(_start), stop(_stop) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::pair<int,int> pos = thrust::get<0>(t);
		thrust::get<1>(t) = (int)(thrust::find(start, stop, pos) - start);
	}
};

struct toRowMajor
{
	int w;
	toRowMajor(int _w) : w(_w) {}

	__host__ __device__
	int operator()(const thrust::pair<int,int> &pos)
	{
		return pos.first + pos.second * w;
	}
};

typedef struct image
{
	char *data;
	int w, h;
}Image;

void initImage(Image *img, int w, int h)
{
	img->w = w;
	img->h = h;
	img->data = (char*)malloc(sizeof(char) * 3 * w * h);
	for(int i = 0; i < w * h; i++)
		img->data[i] = 0;
}

void freeImage(Image *img)
{
	free(img->data);
}

void writeImage(Image *img, char *filename)
{
	FILE* f = fopen(filename, "w");
	fprintf(f, "P6\n%d %d\n255\n", img->w, img->h);
	fwrite(img->data, sizeof(char), 3*img->w*img->h, f);
	fclose(f);
}

int main(int argc, char **argv)
{
	std::cout << "initialize triangles" << std::endl;
	thrust::device_vector<thrust::tuple<float, float, float>> p1(2);
	thrust::device_vector<thrust::tuple<float, float, float>> p2(2);
	thrust::device_vector<thrust::tuple<float, float, float>> p3(2);
	thrust::device_vector<thrust::tuple<char, char, char>> color(2);

	p1[0] = thrust::make_tuple(0,0,0);
	p2[0] = thrust::make_tuple(5,5,0);
	p3[0] = thrust::make_tuple(10,0,0);
	p1[1] = thrust::make_tuple(0,0,-1);
	p2[1] = thrust::make_tuple(10,10,-1);
	p3[1] = thrust::make_tuple(20,0,-1);

	color[0] = thrust::make_tuple(255,0,0);
	color[1] = thrust::make_tuple(0,0,255);

	std::cout << "count fragments" << std::endl;

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

	std::cout << "get fragments" << std::endl;

	thrust::device_vector<int> frag_tri(fragments);
	expand_int(write_index.begin(), frags.begin(), frag_tri.begin(), frag_tri.end(), 2);
	//print_int_vec(frag_tri.begin(), frag_tri.end());
/*
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
	
	//for(int i = 0; i < 2; i++)
	//	std::cout << rows[i] << " ";
	//std::cout << std::endl;

	thrust::device_vector<int> row_off(2);
	thrust::exclusive_scan(rows.begin(), rows.end(), row_off.begin());
	//for(int i = 0; i < 2; i++)
	//	std::cout << row_off[i] << " ";
	//std::cout << std::endl;

	int num_rows = row_off[1] + rows[1];

	thrust::device_vector<int> tri_ptr(num_rows);
	
	expand_int(row_off.begin(), rows.begin(), tri_ptr.begin(), tri_ptr.end(), 2);

	//for(int i = 0; i < num_rows; i++)
	//	std::cout << tri_ptr[i] << " ";
	//std::cout << std::endl;	

	thrust::device_vector<int> row_ptr(num_rows);

	index_int(tri_ptr.begin(), row_off.begin(), row_ptr.begin(), num_rows);

	//print_int_vec(row_ptr.begin(), row_ptr.end());

	thrust::device_vector<int> col_count(num_rows);

	thrust::for_each
		(thrust::make_zip_iterator(thrust::make_tuple
		 	(thrust::make_permutation_iterator(p1.begin(), tri_ptr.begin()),
			 thrust::make_permutation_iterator(p2.begin(), tri_ptr.begin()),
			 thrust::make_permutation_iterator(p3.begin(), tri_ptr.begin()),
			 row_ptr.begin(),
			 col_count.begin())),
		thrust::make_zip_iterator(thrust::make_tuple
		 	(thrust::make_permutation_iterator(p1.begin(), tri_ptr.end()),
			 thrust::make_permutation_iterator(p2.begin(), tri_ptr.end()),
			 thrust::make_permutation_iterator(p3.begin(), tri_ptr.end()),
			 row_ptr.end(),
			 col_count.end())),
		colCount());

	//print_int_vec(col_count.begin(), col_count.end());

	thrust::device_vector<int> col_off(num_rows);

	thrust::exclusive_scan(col_count.begin(), col_count.end(), col_off.begin());

	//print_int_vec(col_off.begin(), col_off.end());

	thrust::device_vector<int> frag_row(fragments);

	expand_int(col_off.begin(), col_count.begin(), frag_row.begin(), frag_row.end(), num_rows);

	thrust::device_vector<int> frag_col(fragments);

	expand_int(col_off.begin(), col_count.begin(), frag_col.begin(), frag_col.end(), num_rows);
	index_int(frag_col.begin(), col_off.begin(), frag_col.begin(), fragments);

	thrust::transform
		(frag_row.begin(),
		 frag_row.end(),
		 thrust::make_permutation_iterator(row_off.begin(), frag_tri.begin()),
		 frag_row.begin(),
		 thrust::minus<int>());

	//print_int_vec(frag_row.begin(), frag_row.end());
	//print_int_vec(frag_col.begin(), frag_col.end());

	thrust::device_vector<thrust::pair<int,int>> pos(fragments);
	thrust::device_vector<float> depth(fragments);

	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::make_permutation_iterator(p1.begin(), frag_tri.begin()),
				thrust::make_permutation_iterator(p2.begin(), frag_tri.begin()),
				thrust::make_permutation_iterator(p3.begin(), frag_tri.begin()),
				frag_row.begin(), frag_col.begin(), pos.begin(), depth.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::make_permutation_iterator(p1.begin(), frag_tri.end()),
				thrust::make_permutation_iterator(p2.begin(), frag_tri.end()),
				thrust::make_permutation_iterator(p3.begin(), frag_tri.end()),
				frag_row.end(), frag_col.end(), pos.end(), depth.end())),
		rasterize());

	//std::cout << "Position and depth of fragments" << std::endl;
	//print_pair_vec(pos.begin(), pos.end());
	//print_float_vec(depth.begin(), depth.end());

	thrust::device_vector<thrust::tuple<char,char,char>> frag_colors(fragments);
	thrust::gather(frag_tri.begin(), frag_tri.end(), color.begin(), frag_colors.begin());
	
	std::cout << "find fragments to write" << std::endl;

	std::cout << "\tcopy position" << std::endl;
	thrust::device_vector<thrust::pair<int,int>> cpos(fragments);
	thrust::device_vector<float> cdepth(fragments);
	thrust::device_vector<thrust::tuple<char,char,char>> cfrag_colors(fragments);
	thrust::device_vector<int> sorted_inds(fragments);
	thrust::sequence(sorted_inds.begin(), sorted_inds.end());

	thrust::copy(pos.begin(), pos.end(), cpos.begin());

	std::cout << "\tsort fragments" << std::endl;

	//std::cout << "Sorted" << std::endl;
	thrust::sort_by_key(cpos.begin(), cpos.end(), sorted_inds.begin());
	thrust::gather(sorted_inds.begin(), sorted_inds.end(), frag_colors.begin(), cfrag_colors.begin());
	thrust::gather(sorted_inds.begin(), sorted_inds.end(), depth.begin(), cdepth.begin());
	//print_pair_vec(cpos.begin(), cpos.end());
	//print_int_vec(sorted_inds.begin(), sorted_inds.end());
	print_float_vec(cdepth.begin(), cdepth.end());

	std::cout << "\tget fragments at lowest depth" << std::endl;
	int unique_positions;
	{
		thrust::device_vector<thrust::pair<int,int>> tmp_pos(fragments);
		auto tmp_pos_end = thrust::unique_copy(cpos.begin(), cpos.end(), tmp_pos.begin());
		unique_positions = (int)(tmp_pos_end - tmp_pos.begin());
	}
	std::cout << "unique positions = " << unique_positions << std::endl;
	thrust::device_vector<thrust::pair<int,int>> true_fragments(unique_positions);
	thrust::device_vector<float> min_depth(unique_positions);
	thrust::device_vector<int> pos_count(unique_positions);
	thrust::reduce_by_key(cpos.begin(), cpos.end(), cdepth.begin(), true_fragments.begin(), 
			min_depth.begin(), thrust::equal_to<thrust::pair<int,int>>(), thrust::maximum<float>());
	thrust::reduce_by_key(cpos.begin(), cpos.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), 
			pos_count.begin(), thrust::equal_to<thrust::pair<int,int>>(), thrust::plus<int>());
	print_int_vec(pos_count.begin(), pos_count.end());
	thrust::device_vector<int> pos_start_ind(unique_positions);
	thrust::exclusive_scan(pos_count.begin(), pos_count.end(), pos_start_ind.begin());
	print_int_vec(pos_start_ind.begin(), pos_start_ind.end());
	thrust::device_vector<int> depth_map(fragments);
	expand_int(pos_start_ind.begin(), pos_count.begin(), depth_map.begin(), depth_map.end(), unique_positions);
	print_int_vec(depth_map.begin(), depth_map.end());
	thrust::device_vector<float> exp_min_depth(fragments);
	thrust::gather(depth_map.begin(), depth_map.end(), min_depth.begin(), exp_min_depth.begin());
	print_float_vec(exp_min_depth.begin(), exp_min_depth.end());
/*
	//std::cout << "Min depth" << std::endl;
	//print_pair_vec(true_fragments.begin(), true_fragments.end());
	//print_float_vec(min_depth.begin(), min_depth.end());

	//thrust::device_vector<thrust::pair<int,int>>::iterator true_end = thrust::unique(true_fragments.begin(), true_fragments.end()) - 1;
	//print_pair_vec(true_fragments.begin(), true_fragments.end());

	std::cout << "\tfor each position, get the shallowest depth of a fragment at that position" << std::endl;
	thrust::device_vector<int> find_real(fragments);
	std::cout <<"\t\tfind each fragment position in list of lowest fragment positions" << std::endl;
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(pos.begin(), find_real.begin())),
			 thrust::make_zip_iterator(thrust::make_tuple(pos.end(), find_real.end())),
			 findPositions(true_fragments.begin(), true_fragments.end()));
	//print_int_vec(find_real.begin(), find_real.end());
	std::cout << "\t\tgather the shallowest depth for each fragment position" << std::endl;
	thrust::device_vector<float> min_depth_by_fragment(fragments);
	thrust::gather(find_real.begin(), find_real.end(), min_depth.begin(), min_depth_by_fragment.begin());
	//std::cout << "Min Depth at fragment position vs fragment depth" << std::endl;
	//print_float_vec(min_depth_by_fragment.begin(), min_depth_by_fragment.end());
	//print_float_vec(depth.begin(), depth.end());

	std::cout << "\tchoose fragments to write" << std::endl;
	thrust::device_vector<bool> write_frag(fragments);
	thrust::transform(min_depth_by_fragment.begin(), min_depth_by_fragment.end(), depth.begin(), write_frag.begin(), thrust::equal_to<float>());
	//std::cout << "Write fragment?" << std::endl;
	//for(int i = 0; i < fragments; i++)
	//	std::cout << write_frag[i] << " ";
	//std::cout << std::endl;
	
	std::cout << "write fragments" << std::endl;

	thrust::device_vector<int> rowMajorPos(fragments);
	thrust::transform(pos.begin(), pos.end(), rowMajorPos.begin(), toRowMajor(width));
	//print_int_vec(rowMajorPos.begin(), rowMajorPos.end());

	thrust::device_vector<thrust::tuple<char,char,char>> img(width * height);
	thrust::fill(img.begin(), img.end(), thrust::make_tuple<char,char,char>(255,255,255));
	thrust::scatter_if(frag_colors.begin(), frag_colors.end(), rowMajorPos.begin(), write_frag.begin(), img.begin());

	thrust::host_vector<thrust::tuple<char,char,char>> h_img = img;

	Image final_image;
	initImage(&final_image, width, height);

	int count = 0;
	for(auto i = h_img.begin(); i < h_img.end(); i++)
	{
		thrust::tuple<char,char,char> t = *i;
		final_image.data[count++] = thrust::get<0>(t);
		final_image.data[count++] = thrust::get<1>(t);
		final_image.data[count++] = thrust::get<2>(t);
	}
	if(argc == 2)
	{
		writeImage(&final_image, argv[1]);
	}
	//char *col = final_image.data;
	//for(int i = 0; i < 60; i+=3)
	//{
	//	std::cout<<(int)col[i]<<","<<(int)col[i+1]<<","<<(int)col[i+2]<<std::endl;
	//}
		
	freeImage(&final_image);
*/
}
