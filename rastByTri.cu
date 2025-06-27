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

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleCast.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleCounting.h>
#include <viskores/cont/ArrayHandleDiscard.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/Invoker.h>
#include <viskores/worklet/ScatterCounting.h>
#include <viskores/worklet/ScatterPermutation.h>
#include <viskores/worklet/WorkletMapField.h>

#include "imageWriter.h"
#include "rastByTri.h"

#ifndef DEBUG
#define DEBUG 0 
#endif

struct ExpandWorklet : viskores::worklet::WorkletMapField
{
	using ControlSignature = void (FieldIn input, FieldIn counts, FieldOut output);
	using ExecutionSignature = void(_1, _3, VisitIndex);
	using InputDomain = _1;

	using ScatterType = viskores::worklet::ScatterCounting;
	
	template<typename T>
	VISKORES_EXEC void operator() (const T &in, T &out, viskores::IdComponent visitIndex) const
	{
		out = in;
	}
};

template<typename PermutationStorage>
struct FillImage : viskores::worklet::WorkletMapField
{
	using ControlSignature = void(FieldIn colors, FieldIn map, FieldIn stencil, WholeArrayOut image);
	using ExecutionSignature = void(_1, _2, _3, _4);

	template<typename InputType, typename StencilType, typename PortalType>
	VISKORES_EXEC
	void operator() (const InputType &col, const viskores::Id &pos, const StencilType &sten, PortalType &img) const
	{
		if (sten) 
		{
			img.Set(pos, col);	
		}
	}	

};

__host__ __device__
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

struct Rasterize : viskores::worklet::WorkletMapField
{
	using ControlSignature = void(
			FieldIn p1,
			FieldIn p2,
			FieldIn p3,
			FieldIn frag_row,
			FieldIn frag_col,
			FieldOut pos,
			FieldOut depth);
	using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);
	template <typename PointType, typename RowColType, typename PositionType, typename DepthType>
	VISKORES_EXEC
	void operator()(const PointType &p1, const PointType &p2, const PointType &p3,
			const RowColType &frag_row, const RowColType &frag_col,
			PositionType &pos, DepthType &depth) const
	{
		float x1, y1, z1, x2, y2, z2, x3, y3, z3;
		x1 = thrust::get<0>(p1);
		y1 = thrust::get<1>(p1);
		z1 = thrust::get<2>(p1);
		x2 = thrust::get<0>(p2);
		y2 = thrust::get<1>(p2);
		z2 = thrust::get<2>(p2);
		x3 = thrust::get<0>(p3);
		y3 = thrust::get<1>(p3);
		z3 = thrust::get<2>(p3);
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
		float y = ceil(minY) + frag_row;
		float end1, end2;
		getEnds(x1,y1,x2,y2,x3,y3,y,end1,end2);
		int x = ceil(end1) + frag_col;
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
		pos = thrust::make_pair(x, y);
		depth = z;
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

/*
    Given some list of groups let pred be the
    number of elements in each group, and offset
    be the starting position of each group in a
    list of all elements in a supergroup containing
    all groups. expand_int generates a list of all elements
    of the supergroup where the value at an elements indice
    is the index of the group it belongs to.
*/
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

/*
   Take a list of values and a list of counts,
   and duplicate each value a number of times
   equal to the count at its index
*/
template<typename T, typename CountT>
void vduplicate(const viskores::cont::ArrayHandle<T> &values,
		 const viskores::cont::ArrayHandle<CountT> &count,
		 viskores::cont::ArrayHandle<T> &output)
{
	viskores::cont::Invoker invoke;
	viskores::worklet::ScatterCounting scatter(count);
	ExpandWorklet expand_worklet;
	invoke(
		expand_worklet,
		scatter,
		values,
		count,
		output
	);

}

/*
    Given some list of groups let counts be the
    number of elements in each group, and assume
    that the order in which groups appear in counts
    is the same as the order in which they appear in
    a supergroup containing all groups. vexpand generates
    a list of all elements of the supergroup where
    the value at an element's indice is the index 
    of the group it belongs to.
*/
template<typename T, typename CountT>
void vexpand(viskores::cont::ArrayHandle<CountT> &counts,
		 viskores::cont::ArrayHandle<T> &output)
{
	viskores::Id length = counts.GetNumberOfValues();
	viskores::cont::ArrayHandle<T> sequence;
	viskores::cont::ArrayCopy
		(viskores::cont::make_ArrayHandleCounting<T>(0, 1, length),
		 sequence);
	vduplicate<T, CountT>(
		sequence,
		counts,
		output
	);
}

/*
   Let map be a list associating elements to their
   groups, and src be a list of group offsets.
   index_int generates a list where the value at an
   element's index is its index within its group.
*/
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

/*
   Let map be a list associating elements to their
   groups, and src be a list of group offsets.
   vindex generates a list where the value at an
   element's index is its index within its group.
*/
template<typename IndexType, typename ValueType>
void vindex
	(viskores::cont::ArrayHandle<IndexType> &map,
	 viskores::cont::ArrayHandle<ValueType> &src,
	 viskores::cont::ArrayHandle<ValueType> &out)
{
	viskores::Id length = map.GetNumberOfValues();
	viskores::cont::Algorithm::Transform
		(viskores::cont::make_ArrayHandleCounting<ValueType>(0, 1, length),
		 viskores::cont::make_ArrayHandlePermutation(map, src),
		 out, thrust::minus<ValueType>());
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

template<typename T>
void print_ArrayHandle(const viskores::cont::ArrayHandle<T> &arr)
{
	auto arr_Reader = arr.ReadPortal();
	for (viskores::Id i = 0; i < arr_Reader.GetNumberOfValues(); i++)
	{
		std::cout << arr_Reader.Get(i) << "\t";
	}
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

struct ToRowMajor : viskores::worklet::WorkletMapField
{
	int w;
	ToRowMajor(int _w) : w(_w) {}

	using ControlSignature = void(FieldIn coordinates, FieldOut indices);
	using ExecutionSignature = _2(_1);

	VISKORES_EXEC
	int operator()(const thrust::pair<int,int> &pos) const
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
	
	//Define a Viskores Invoker
	viskores::cont::Invoker invoke;

/*
   RASTERIZE
*/

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
	//Copy vectors to ArrayHandles
	viskores::cont::ArrayHandle<int> vcol_count = 
		viskores::cont::make_ArrayHandle(thrust::raw_pointer_cast(col_count.data()), col_count.size(), viskores::CopyFlag::On);

	viskores::cont::ArrayHandle<viskores::Id> vcol_off;

	viskores::cont::Algorithm::ScanExclusive
		(viskores::cont::make_ArrayHandleCast<viskores::Id>(vcol_count), vcol_off);
#if DEBUG > 2 
	std::cout << "Column offsets by row" << std::endl;
	print_int_vec(col_off.begin(), col_off.end());
	std::cout << "Number of columns " <<  col_off[num_rows-1] + col_count[num_rows-1] << std::endl;
#endif
	assert((fragments == (int)vcol_off.ReadPortal().Get(num_rows-1) + (int)vcol_count.ReadPortal().Get(num_rows-1)));
	//Copy vectors to array handles
	std::vector<viskores::Id> tmp_frag_tri(frag_tri.begin(), frag_tri.end());
	viskores::cont::ArrayHandle<viskores::Id> vfrag_tri = 
		viskores::cont::make_ArrayHandle(tmp_frag_tri, viskores::CopyFlag::On);
	std::vector<viskores::Id> tmp_row_off(row_off.begin(), row_off.end());
	viskores::cont::ArrayHandle<viskores::Id> vrow_off = 
		viskores::cont::make_ArrayHandle(tmp_row_off, viskores::CopyFlag::On);

	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<viskores::Id> vfrag_row;
	viskores::cont::ArrayHandle<viskores::Id> vfrag_col;

	//Determine fragment rows and columns
	vexpand(vcol_count, vfrag_row);
	std::cout << "Frag Rows" << std::endl;
	//print_ArrayHandle(vfrag_row);

	//temporary copies
	/*
	viskores::cont::ArrayHandle<viskores::Id> vtmp_frag_row;
	viskores::cont::ArrayHandle<viskores::Id> vtmp_frag_col;
	vtmp_frag_row.DeepCopyFrom(vfrag_row);
	vtmp_frag_col.DeepCopyFrom(vfrag_col);
	*/

	vindex(vfrag_row, vcol_off, vfrag_col);

	viskores::cont::Algorithm::Transform
		(vfrag_row,
		 viskores::cont::make_ArrayHandlePermutation(vfrag_tri, vrow_off),
		 vfrag_row,
		 thrust::minus<viskores::Id>());
	//std::cout << "Size of frag_row, frag_col: " <<
	//	vfrag_row.GetNumberOfValues() << ", " <<
	//	vfrag_col.GetNumberOfValues() << std::endl;
	//std::cout << "Frag Col" << std::endl;
	//print_ArrayHandle(vfrag_col);
		 
#if DEBUG > 3 
	std::cout << "Frag positions by row and column in every triangle." << std::endl;
	print_int_vec(frag_row.begin(), frag_row.end());
	print_int_vec(frag_col.begin(), frag_col.end());
#endif
	//Copy vectors to ArrayHandles
	viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> vp1 = 
		viskores::cont::make_ArrayHandle(thrust::raw_pointer_cast(p1.data()), p1.size(), viskores::CopyFlag::On);
	viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> vp2 = 
		viskores::cont::make_ArrayHandle(thrust::raw_pointer_cast(p2.data()), p2.size(), viskores::CopyFlag::On);
	viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> vp3 = 
		viskores::cont::make_ArrayHandle(thrust::raw_pointer_cast(p3.data()), p3.size(), viskores::CopyFlag::On);

	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<thrust::pair<int,int>> vpos;
	viskores::cont::ArrayHandle<float> vdepth;

	/*thrust::for_each(
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
		rasterize());*/

	//Rasterize
	Rasterize rasterize;	
	invoke(rasterize,
		viskores::cont::make_ArrayHandlePermutation(vfrag_tri, vp1),
		viskores::cont::make_ArrayHandlePermutation(vfrag_tri, vp2),
		viskores::cont::make_ArrayHandlePermutation(vfrag_tri, vp3),
		vfrag_row, vfrag_col, vpos, vdepth);

	/*auto tmp_pos_Reader = vpos.ReadPortal();
	for(viskores::Id i = 0; i < tmp_pos_Reader.GetNumberOfValues(); i++)
	{
		std::cout << "(" << thrust::get<0>(tmp_pos_Reader.Get(i)) << ", " << 
			thrust::get<1>(tmp_pos_Reader.Get(i)) << ")\t";
	}
	std::cout << std::endl;
	auto tmp_dep_Reader = vdepth.ReadPortal();
	for(viskores::Id i = 0; i < tmp_dep_Reader.GetNumberOfValues(); i++)
	{
		std::cout << tmp_dep_Reader.Get(i) << "\t";
	}
	std::cout << std::endl;*/

#if DEBUG > 3
	std::cout << "Position and depth of fragments" << std::endl;
	print_pair_vec(pos.begin(), pos.end());
	print_float_vec(depth.begin(), depth.end());
#endif
	//Copy vectors to ArrayHandles
	viskores::cont::ArrayHandle<thrust::tuple<char,char,char>> vcolor = 
		viskores::cont::make_ArrayHandle(thrust::raw_pointer_cast(color.data()), color.size(), viskores::CopyFlag::On);

	//Gather the color of each fragment
	viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandle<thrust::tuple<char,char,char>>> vfrag_colors(vfrag_tri, vcolor);

	//time: rasterized triangles. acquired all fragments
	timer.push_back(std::chrono::high_resolution_clock::now());

/*
   SORT
*/

#if DEBUG > 0	
	std::cout << "find fragments to write" << std::endl;

	std::cout << "\tcopy position" << std::endl;
#endif

	//Allocate ArrayHandles for Sorting
	viskores::cont::ArrayHandle<thrust::pair<int, int>> vcpos;
	vcpos.DeepCopyFrom(vpos);	
	viskores::cont::ArrayHandleCounting<viskores::Id> tmp_inds(0, 1, fragments);
	viskores::cont::ArrayHandle<viskores::Id> vsorted_inds;
	viskores::cont::Algorithm::Copy(tmp_inds, vsorted_inds);

#if DEBUG > 0
	std::cout << "\tsort fragments" << std::endl;
#endif
	viskores::cont::Algorithm::SortByKey(vcpos, vsorted_inds);
	viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandle<thrust::tuple<char,char,char>>>> vcfrag_colors(vsorted_inds, vfrag_colors);
	viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandle<float>> vcdepth(vsorted_inds, vdepth);

#if DEBUG > 3
	std::cout << "Sorted" << std::endl;
	print_pair_vec(cpos.begin(), cpos.end());
	print_int_vec(sorted_inds.begin(), sorted_inds.end());
	print_float_vec(cdepth.begin(), cdepth.end());
#endif
	//time: sorted fragments
	timer.push_back(std::chrono::high_resolution_clock::now());

/*
   SELECT
*/

#if DEBUG > 0
	std::cout << "\tget fragments at lowest depth" << std::endl;
#endif
/*
	//count the number of unique positions
	int unique_positions;
	{
		viskores::cont::ArrayHandle<thrust::pair<int,int>> tmp_pos;
		viskores::cont::Algorithm::Copy(vcpos, tmp_pos);
		viskores::cont::Algorithm::Unique(tmp_pos);
		unique_positions = tmp_pos.GetNumberOfValues();
	}
#if DEBUG > 1
	std::cout << "\tunique positions = " << unique_positions << std::endl;
#endif
*/
	viskores::cont::ArrayHandle<thrust::pair<int,int>> vtrue_fragments;
	viskores::cont::ArrayHandle<float> vmin_depth;
	viskores::cont::ArrayHandle<int> vpos_count;
	viskores::cont::Algorithm::ReduceByKey(vcpos, vcdepth, vtrue_fragments, vmin_depth, thrust::maximum<float>());
	viskores::cont::Algorithm::ReduceByKey(vcpos, viskores::cont::make_ArrayHandleConstant<int>(1, fragments),
		       vtrue_fragments, vpos_count, thrust::plus<int>());	

#if DEBUG > 3
	std::cout << "Number of duplicates at each unique position" << std::endl;
	print_int_vec(pos_count.begin(), pos_count.end());
#endif
#if DEBUG > 0
	std::cout << "\tGet the minimum depth of each unique position" << std::endl;
#endif
	/* Thrust Implementation

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

	*/

	/* Viskores Implementation */

	viskores::cont::ArrayHandle<float> vexp_min_depth;
	vduplicate(vmin_depth, vpos_count, vexp_min_depth);

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
	viskores::cont::ArrayHandle<bool> vwrite_frag;
	viskores::cont::Algorithm::Transform(vexp_min_depth, vcdepth, vwrite_frag, thrust::equal_to<float>());

	/*
	//Convert ArrayHandles to Thrust vectors

	//Create portals for reading
	auto pos_Reader = vcpos.ReadPortal();
	auto depth_Reader = vcdepth.ReadPortal();
	auto color_Reader = vcfrag_colors.ReadPortal();
	auto inds_Reader = vsorted_inds.ReadPortal();

	auto true_frag_reader = vtrue_fragments.ReadPortal();
	auto min_depth_reader = vmin_depth.ReadPortal();
	auto pos_count_reader = vpos_count.ReadPortal();
	auto exp_min_depth_reader = vexp_min_depth.ReadPortal();
	auto write_frag_reader = vwrite_frag.ReadPortal();

	//Create Thrust vectors
	thrust::device_vector<thrust::pair<int,int>> cpos(
		viskores::cont::ArrayPortalToIteratorBegin(pos_Reader),
		viskores::cont::ArrayPortalToIteratorEnd(pos_Reader)
	);
	thrust::device_vector<float> cdepth(
		viskores::cont::ArrayPortalToIteratorBegin(depth_Reader),
		viskores::cont::ArrayPortalToIteratorEnd(depth_Reader)
	);
	thrust::device_vector<thrust::tuple<char,char,char>> cfrag_colors(
		viskores::cont::ArrayPortalToIteratorBegin(color_Reader),
		viskores::cont::ArrayPortalToIteratorEnd(color_Reader)
	);
	thrust::device_vector<int> sorted_inds(
		viskores::cont::ArrayPortalToIteratorBegin(inds_Reader),
		viskores::cont::ArrayPortalToIteratorEnd(inds_Reader)
	);

	thrust::device_vector<thrust::pair<int,int>> true_fragments(
		viskores::cont::ArrayPortalToIteratorBegin(true_frag_reader),
		viskores::cont::ArrayPortalToIteratorEnd(true_frag_reader)
	);
	thrust::device_vector<float> min_depth(
		viskores::cont::ArrayPortalToIteratorBegin(min_depth_reader),
		viskores::cont::ArrayPortalToIteratorEnd(min_depth_reader)
	);
	thrust::device_vector<int> pos_count(
		viskores::cont::ArrayPortalToIteratorBegin(pos_count_reader),
		viskores::cont::ArrayPortalToIteratorEnd(pos_count_reader)
	);
	thrust::device_vector<float> exp_min_depth(
		viskores::cont::ArrayPortalToIteratorBegin(exp_min_depth_reader),
		viskores::cont::ArrayPortalToIteratorEnd(exp_min_depth_reader)
	);
	thrust::device_vector<bool> write_frag(
		viskores::cont::ArrayPortalToIteratorBegin(write_frag_reader),
		viskores::cont::ArrayPortalToIteratorEnd(write_frag_reader)
	);
	*/

#if DEBUG > 3
	std::cout << "Write fragment?" << std::endl;
	for(int i = 0; i < fragments; i++)
		std::cout << write_frag[i] << " ";
	std::cout << std::endl;
#endif
	//time: got visible fragments
	timer.push_back(std::chrono::high_resolution_clock::now());

/*
   WRITE
*/

#if DEBUG > 0
	std::cout << "write fragments" << std::endl;
#endif

	viskores::cont::ArrayHandle<viskores::Id> vrowMajorPos;
	ToRowMajor to_row_major(width);
	invoke(to_row_major, vcpos, vrowMajorPos);

#if DEBUG > 3
	std::cout << "Row major position by fragment" << std::endl;
	print_int_vec(rowMajorPos.begin(), rowMajorPos.end());
#endif

	//viskores::cont::ArrayHandle<thrust::tuple<char,char,char>> vbg;
	//vbg.AllocateAndFill(width * height, thrust::make_tuple<char,char,char>(127,127,127));
	viskores::cont::ArrayHandle<thrust::tuple<char,char,char>> vimg;
	vimg.AllocateAndFill(width * height, thrust::make_tuple<char,char,char>(127,127,127));
	/*
	std::cout << vcfrag_colors.GetNumberOfValues() << std::endl;
	std::cout << vrowMajorPos.GetNumberOfValues() << std::endl;
	std::cout << vwrite_frag.GetNumberOfValues() << std::endl;
	std::cout << vimg.GetNumberOfValues() << std::endl;
	*/
	//auto max_pos = viskores::cont::Algorithm::Reduce(vrowMajorPos, (viskores::Id) 0,
	//	       [](const auto& a, const auto& b){return std::max(a,b);});	
	//std::cout << max_pos << std::endl;
	FillImage<viskores::cont::StorageTagBasic> fill_image;
	invoke(
		fill_image,
		vcfrag_colors,
		vrowMajorPos,
		vwrite_frag,
		vimg
	);

	auto img_Reader = vimg.ReadPortal();
	int count = 0;
	for(viskores::Id i = 0; i < img_Reader.GetNumberOfValues(); i++)
	{
		thrust::tuple<char,char,char> t = img_Reader.Get(i);
		final_image.data[count++] = thrust::get<0>(t);
		final_image.data[count++] = thrust::get<1>(t);
		final_image.data[count++] = thrust::get<2>(t);
	}
	//time: write final image to output
	timer.push_back(std::chrono::high_resolution_clock::now());

/*
   DONE
*/

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
