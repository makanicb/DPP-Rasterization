#include <iostream>
#include <cmath>
#include <cassert>
#include <limits>
#include <chrono>

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

#include "imageWriter.h"
#include "rastByTri.h"

#ifndef DEBUG
#define DEBUG 0 
#endif

void getEnds(float x1, float y1, float x2, float y2, float x3, float y3, float y, float &end1, float &end2)
{
		float ed1, ed2, ed3;
		bool e1 = false, e2 = false, e3 = false;
		if((y1 < y2 && y >= y1 && y <= y2) || (y1 > y2 && y >= y2 && y <= y1)) 
		{
			ed1 = (y - y1) * (x2-x1) / (y2-y1) + x1;
			e1 = true;
		}
		if((y2 < y3 && y >= y2 && y <= y3) || (y2 > y3 && y >= y3 && y <= y2)) 
		{
			ed2 = (y - y2) * (x3-x2) / (y3-y2) + x2;
			e2 = true;
		}
		if((y1 < y3 && y >= y1 && y <= y3) || (y1 > y3 && y >= y3 && y <= y1)) 
		{
			ed3 = (y - y1) * (x3-x1) / (y3-y1) + x1;
			e3 = true;
		}

		if(y1 == y2 && y2 == y3 && y3 == y)
		{
			end1 = x1 < x2 ? x1 : x2;
			end1 = end1 < x3 ? end1 : x3;
			end2 = x1 < x2 ? x2 : x1;
			end2 = end2 < x3 ? x3 : end2;
		}
		else if(e1 && e2 && e3)
		{
			end1 = ed1 < ed2 ? ed1 : ed2;
			end1 = end1 < ed3 ? end1 : ed3;
			end2 = ed1 < ed2 ? ed2 : ed1;
			end2 = end2 < ed3 ? ed3 : end2;
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
}

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
		float minY = y1 < y2 ? y1 : y2;
		minY = minY < y3 ? minY : y3;
		float maxY = y1 > y2 ? y1 : y2;
		maxY = maxY > y3 ? maxY : y3;
		int low = ceil(minY);
		int high = floor(maxY);
		int frags = 0;
		for(int i = low; i <= high; i++)
		{
			float end1, end2;
			getEnds(x1,y1,x2,y2,x3,y3,i,end1,end2);
			frags += floor(end2) - ceil(end1) + 1;
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
		/*std::cout << x1 << ","
			  << y1 << ","
			  << z1 << std::endl;
		std::cout << x2 << ","
			  << y2 << ","
			  << z2 << std::endl;
		std::cout << x3 << ","
			  << y3 << ","
			  << z3 << std::endl;*/
		//calculate triangle plane
		float minY = y1 < y2 ? y1 : y2;
		minY = minY < y3 ? minY : y3;
		float y = ceil(minY) + thrust::get<3>(t);
		float end1, end2;
		getEnds(x1,y1,x2,y2,x3,y3,y,end1,end2);

		int x = ceil(end1) + thrust::get<4>(t);
		float z;
		float x_coe = ((y2-y1)*(z3-z1)-(y3-y1)*(z2-z1));
		float y_coe = ((x2-x1)*(z3-z1)-(x3-x1)*(z2-z1));
		float z_coe = ((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1));
		//z_coe is zero if and only if (x2-x1)*(y3-y1)==(x3-x1)*(y2-y1)
		//Then if z_coe is zero then the triangle is a line on the xy plane
		if(z_coe){
			z = z1 - (x_coe*(x-x1)+y_coe*(y1-y))/z_coe;
		}else if(y1 != y2 || y2 != y3 || y1 != y3){
			float minZ, maxZ; 
			getEnds(z1,y1,z2,y2,z3,y3,y,minZ,maxZ);
			z = maxZ;
		}else if(x1 != x2 || x2 != x3 || x1 != x3){
			float minZ, maxZ;
			getEnds(z1,x1,z2,x2,z3,x3,x,minZ,maxZ);
			z = maxZ;
		}else{
			z = z1 > z2 ? z1 : z2;
			z = z > z3 ? z : z3;
		}
		thrust::get<5>(t)  = thrust::make_pair(x, y);
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
		float minY = y1 < y2 ? y1 : y2; 
		minY = minY < y3 ? minY : y3;
		float maxY = y1 > y2 ? y1 : y2; 
		maxY = maxY > y3 ? maxY : y3;
		thrust::get<3>(t) = floor(maxY) - ceil(minY) + 1; 
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
		float minY = y1 < y2 ? y1 : y2;
		minY = minY < y3 ? minY : y3;
		float y = ceil(minY) + row;
		float end1, end2;
		getEnds(x1,y1,x2,y2,x3,y3,y,end1,end2);
		thrust::get<4>(t) = floor(end2) - ceil(end1) + 1;
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

void RasterizeTriangles(thrust::device_vector<thrust::tuple<float, float, float>> &p1,
		thrust::device_vector<thrust::tuple<float, float, float>> &p2,
		thrust::device_vector<thrust::tuple<float, float, float>> &p3,
		thrust::device_vector<thrust::tuple<char, char, char>> &color,
		int numTri, int width, int height, Image &final_image)
{
	//Set up timing systems
	thrust::host_vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timer;
	//time: function start
	timer.push_back(std::chrono::high_resolution_clock::now());	
#if DEBUG > 0
	std::cout << "Count fragments" << std::endl;
#endif
#if DEBUG > 1 
	std::cout << numTri << " Triangles" << std::endl;
#endif	
	thrust::device_vector<int> frags(numTri);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(p1.begin(), p2.begin(), p3.begin(), frags.begin())),
			 thrust::make_zip_iterator(thrust::make_tuple(p1.end(), p2.end(), p3.end(), frags.end())),
			 fragCount());
#if DEBUG > 1 
	std::cout << "# frags by triange: " << std::endl;
	print_int_vec(frags.begin(), frags.end());
#endif

	thrust::device_vector<int> write_index(numTri);

	thrust::exclusive_scan(frags.begin(), frags.end(), write_index.begin());
#if DEBUG > 1
	std::cout << "write position by triange: " << std::endl;
	print_int_vec(write_index.begin(), write_index.end());
#endif

	int fragments = write_index[numTri-1] + frags[numTri-1];
#if DEBUG > 1	
	std::cout << "Number of fragments: " << fragments << std::endl;
#endif
#if DEBUG > 0
	std::cout << "Get fragments" << std::endl;
#endif

	thrust::device_vector<int> frag_tri(fragments);
	expand_int(write_index.begin(), frags.begin(), frag_tri.begin(), frag_tri.end(), numTri);
#if DEBUG > 3
	std::cout << "Which triangle does each fragment belong to?" << std::endl;
	print_int_vec(frag_tri.begin(), frag_tri.end());
#endif
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
		 thrust::make_permutation_iterator(write_index.begin(), frag_pos.begin()), frag_ind.begin(),
		 thrust::minus<int>());
*/	
	thrust::device_vector<int> rows(numTri);
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(p1.begin(), p2.begin(), p3.begin(), rows.begin())),
			 thrust::make_zip_iterator(thrust::make_tuple(p1.end(), p2.end(), p3.end(), rows.end())),
			 rowCount());
#if DEBUG > 1
	std::cout << "How many rows does each triangle have?" << std::endl;
	for(int i = 0; i < numTri; i++)
		std::cout << rows[i] << " ";
	std::cout << std::endl;
#endif

	thrust::device_vector<int> row_off(numTri);
	thrust::exclusive_scan(rows.begin(), rows.end(), row_off.begin());
#if DEBUG > 1
	std::cout << "What is the row offset of each triangle?" << std::endl;
	for(int i = 0; i < numTri; i++)
		std::cout << row_off[i] << " ";
	std::cout << std::endl;
#endif

	int num_rows = row_off[numTri-1] + rows[numTri-1];

	thrust::device_vector<int> tri_ptr(num_rows);
	
	expand_int(row_off.begin(), rows.begin(), tri_ptr.begin(), tri_ptr.end(), numTri);
#if DEBUG > 2 
	std::cout << "What triangle does each row belong to?" << std::endl;
	for(int i = 0; i < num_rows; i++)
		std::cout << tri_ptr[i] << " ";
	std::cout << std::endl;	
#endif

	thrust::device_vector<int> row_ptr(num_rows);

	index_int(tri_ptr.begin(), row_off.begin(), row_ptr.begin(), num_rows);
#if DEBUG > 2 
	std::cout << "The index of each row." << std::endl;
	print_int_vec(row_ptr.begin(), row_ptr.end());
#endif

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
#if DEBUG > 2
	std::cout << "How many columns does each row have?" << std::endl;
	print_int_vec(col_count.begin(), col_count.end());
#endif

	thrust::device_vector<int> col_off(num_rows);

	thrust::exclusive_scan(col_count.begin(), col_count.end(), col_off.begin());
#if DEBUG > 2 
	std::cout << "Column offsets by row" << std::endl;
	print_int_vec(col_off.begin(), col_off.end());
	std::cout << "Number of columns " <<  col_off[num_rows-1] + col_count[num_rows-1] << std::endl;
#endif
	assert((fragments == (int)col_off[num_rows-1] + (int)col_count[num_rows-1]));
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
#if DEBUG > 3 
	std::cout << "Frag positions by row and column in every triangle." << std::endl;
	print_int_vec(frag_row.begin(), frag_row.end());
	print_int_vec(frag_col.begin(), frag_col.end());
#endif

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
#if DEBUG > 3
	std::cout << "Position and depth of fragments" << std::endl;
	print_pair_vec(pos.begin(), pos.end());
	print_float_vec(depth.begin(), depth.end());
#endif

	thrust::device_vector<thrust::tuple<char,char,char>> frag_colors(fragments);
	thrust::gather(frag_tri.begin(), frag_tri.end(), color.begin(), frag_colors.begin());
	//time: rasterized triangles. acquired all fragments
	timer.push_back(std::chrono::high_resolution_clock::now());
#if DEBUG > 0	
	std::cout << "find fragments to write" << std::endl;

	std::cout << "\tcopy position" << std::endl;
#endif
	thrust::device_vector<thrust::pair<int,int>> cpos(fragments);
	thrust::device_vector<float> cdepth(fragments);
	thrust::device_vector<thrust::tuple<char,char,char>> cfrag_colors(fragments);
	thrust::device_vector<int> sorted_inds(fragments);
	thrust::sequence(sorted_inds.begin(), sorted_inds.end());

	thrust::copy(pos.begin(), pos.end(), cpos.begin());
#if DEBUG > 0
	std::cout << "\tsort fragments" << std::endl;
#endif
	thrust::sort_by_key(cpos.begin(), cpos.end(), sorted_inds.begin());
	thrust::gather(sorted_inds.begin(), sorted_inds.end(), frag_colors.begin(), cfrag_colors.begin());
	thrust::gather(sorted_inds.begin(), sorted_inds.end(), depth.begin(), cdepth.begin());
#if DEBUG > 3
	std::cout << "Sorted" << std::endl;
	print_pair_vec(cpos.begin(), cpos.end());
	print_int_vec(sorted_inds.begin(), sorted_inds.end());
	print_float_vec(cdepth.begin(), cdepth.end());
#endif
	//time: sorted fragments
	timer.push_back(std::chrono::high_resolution_clock::now());
#if DEBUG > 0
	std::cout << "\tget fragments at lowest depth" << std::endl;
#endif
	int unique_positions;
	{
		thrust::device_vector<thrust::pair<int,int>> tmp_pos(fragments);
		auto tmp_pos_end = thrust::unique_copy(cpos.begin(), cpos.end(), tmp_pos.begin());
		unique_positions = (int)(tmp_pos_end - tmp_pos.begin());
	}
#if DEBUG > 1
	std::cout << "\tunique positions = " << unique_positions << std::endl;
#endif
	thrust::device_vector<thrust::pair<int,int>> true_fragments(unique_positions);
	thrust::device_vector<float> min_depth(unique_positions);
	thrust::device_vector<int> pos_count(unique_positions);
	thrust::reduce_by_key(cpos.begin(), cpos.end(), cdepth.begin(), true_fragments.begin(), 
			min_depth.begin(), thrust::equal_to<thrust::pair<int,int>>(), thrust::maximum<float>());
	thrust::reduce_by_key(cpos.begin(), cpos.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), 
			pos_count.begin(), thrust::equal_to<thrust::pair<int,int>>(), thrust::plus<int>());
#if DEBUG > 3
	std::cout << "Number of duplicates at each unique position" << std::endl;
	print_int_vec(pos_count.begin(), pos_count.end());
#endif
#if DEBUG > 0
	std::cout << "\tGet the minimum depth of each unique position" << std::endl;
#endif
	thrust::device_vector<int> pos_start_ind(unique_positions);
	thrust::exclusive_scan(pos_count.begin(), pos_count.end(), pos_start_ind.begin());
#if DEBUG > 3
	std::cout << "Offset by unique position" << std::endl;
	print_int_vec(pos_start_ind.begin(), pos_start_ind.end());
#endif
	thrust::device_vector<int> depth_map(fragments);
	expand_int(pos_start_ind.begin(), pos_count.begin(), depth_map.begin(), depth_map.end(), unique_positions);
#if DEBUG > 3
	std::cout << "Min depth gather position by fragment" << std::endl;
	print_int_vec(depth_map.begin(), depth_map.end());
#endif
	thrust::device_vector<float> exp_min_depth(fragments);
	thrust::gather(depth_map.begin(), depth_map.end(), min_depth.begin(), exp_min_depth.begin());
#if DEBUG > 3
	std::cout << "Min depth by fragment" << std::endl;
	print_float_vec(exp_min_depth.begin(), exp_min_depth.end());
#endif
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
*/
#if DEBUG > 3
	std::cout << "Min Depth at fragment position vs fragment depth" << std::endl;
	print_float_vec(exp_min_depth.begin(), exp_min_depth.end());
	print_float_vec(cdepth.begin(), cdepth.end());
#endif
#if DEBUG > 0
	std::cout << "\tchoose fragments to write" << std::endl;
#endif
	thrust::device_vector<bool> write_frag(fragments);
	thrust::transform(exp_min_depth.begin(), exp_min_depth.end(), cdepth.begin(), write_frag.begin(), thrust::equal_to<float>());
#if DEBUG > 3
	std::cout << "Write fragment?" << std::endl;
	for(int i = 0; i < fragments; i++)
		std::cout << write_frag[i] << " ";
	std::cout << std::endl;
#endif
	//time: got visible fragments
	timer.push_back(std::chrono::high_resolution_clock::now());
#if DEBUG > 0
	std::cout << "write fragments" << std::endl;
#endif

	thrust::device_vector<int> rowMajorPos(fragments);
	thrust::transform(cpos.begin(), cpos.end(), rowMajorPos.begin(), toRowMajor(width));
#if DEBUG > 3
	std::cout << "Row major position by fragment" << std::endl;
	print_int_vec(rowMajorPos.begin(), rowMajorPos.end());
#endif

	thrust::device_vector<thrust::tuple<char,char,char>> img(width * height);
	thrust::fill(img.begin(), img.end(), thrust::make_tuple<char,char,char>(255,255,255));
	thrust::scatter_if(cfrag_colors.begin(), cfrag_colors.end(), rowMajorPos.begin(), write_frag.begin(), img.begin());

	thrust::host_vector<thrust::tuple<char,char,char>> h_img = img;
	
	int count = 0;
	for(auto i = h_img.begin(); i < h_img.end(); i++)
	{
		thrust::tuple<char,char,char> t = *i;
		final_image.data[count++] = thrust::get<0>(t);
		final_image.data[count++] = thrust::get<1>(t);
		final_image.data[count++] = thrust::get<2>(t);
	}
	//time: write final image to output
	timer.push_back(std::chrono::high_resolution_clock::now());
	//char *col = final_image.data;
	//for(int i = 0; i < 60; i+=3)
	//{
	//	std::cout<<(int)col[i]<<","<<(int)col[i+1]<<","<<(int)col[i+2]<<std::endl;
	//}
	auto p = timer.begin();
	for(auto i = timer.begin() + 1; i != timer.end(); i++)
	{
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(*i - *p);
		p = i;
		std::cout << "\t" << duration.count();	
	}
	std::cout << std::endl;
}
