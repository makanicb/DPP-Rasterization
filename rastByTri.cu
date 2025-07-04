#include <iostream>
#include <cmath>
#include <cassert>
#include <limits>
#include <chrono>

/*
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
*/

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleCast.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleCounting.h>
#include <viskores/cont/ArrayHandleDiscard.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/Invoker.h>
#include <viskores/Types.h>
#include <viskores/worklet/ScatterCounting.h>
#include <viskores/worklet/ScatterPermutation.h>
#include <viskores/worklet/WorkletMapField.h>

#include "imageWriter.h"
#include "rastByTri.h"

#ifndef DEBUG
#define DEBUG 0 
#endif

template<typename T>
struct my_maximum
{
	T operator()(const T &lhs, const T &rhs) const
	{
		return lhs > rhs ? lhs : rhs;
	}
};

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

VISKORES_EXEC
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

struct FragCount : viskores::worklet::WorkletMapField
{
	using ControlSignature = void(
		FieldIn p1,
		FieldIn p2,
		FieldIn p3,
		FieldOut fragCount);
	using ExecutionSignature = void(_1, _2, _3, _4);
	template <typename PointType, typename FragCountType>
	VISKORES_EXEC
	void operator()(const PointType &p1, const PointType &p2, const PointType &p3,
			FragCountType &fragCount) const
	{
		float x1, y1, x2, y2, x3, y3;
		x1 = p1[0];
		y1 = p1[1];
		x2 = p2[0];
		y2 = p2[1];
		x3 = p3[0];
		y3 = p3[1];
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

		fragCount = frags;
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
		x1 = p1[0];
		y1 = p1[1];
		z1 = p1[2];
		x2 = p2[0];
		y2 = p2[1];
		z2 = p2[2];
		x3 = p3[0];
		y3 = p3[1];
		z3 = p3[2];
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
		pos = viskores::make_Vec(x, y);
		depth = z;
	}
};

struct RowCount : viskores::worklet::WorkletMapField
{
	using ControlSignature = void(
		FieldIn p1,
		FieldIn p2,
		FieldIn p3,
		FieldOut rowCount
	);
	using ExecutionSignature = void(_1, _2, _3, _4);
	template <typename PointType, typename RowCountType>
	VISKORES_EXEC
	void operator()(const PointType &p1, const PointType &p2, const PointType &p3,
			RowCountType &rowCount) const
	{
		float y1, y2, y3;
		y1 = p1[1];
		y2 = p2[1];
		y3 = p3[1]; 
		float minY = y1 < y2 ? y1 : y2; 
		minY = minY < y3 ? minY : y3;
		float maxY = y1 > y2 ? y1 : y2; 
		maxY = maxY > y3 ? maxY : y3;
		rowCount = floor(maxY) - ceil(minY) + 1; 
	}
};

struct ColCount : viskores::worklet::WorkletMapField
{
	using ControlSignature = void(
		FieldIn p1,
		FieldIn p2,
		FieldIn p3,
		FieldIn row,
		FieldOut colCount);
	using ExecutionSignature = void(_1, _2, _3, _4, _5);

	template <typename PointType, typename RowType, typename ColCountType>
	VISKORES_EXEC
	void operator()(const PointType &p1, const PointType &p2, const PointType &p3,
			const RowType &row, ColCountType &colCount) const
	{
		float x1, y1, x2, y2, x3, y3;
		x1 = p1[0];
		y1 = p1[1];
		x2 = p2[0];
		y2 = p2[1];
		x3 = p3[0];
		y3 = p3[1];
		float minY = y1 < y2 ? y1 : y2;
		minY = minY < y3 ? minY : y3;
		float y = ceil(minY) + row;
		float end1, end2;
		getEnds(x1,y1,x2,y2,x3,y3,y,end1,end2);
		colCount = floor(end2) - ceil(end1) + 1;
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
/*
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
*/

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
/*
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
*/

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
		 out, std::minus<ValueType>());
}

/*
void print_int_vec(thrust::device_vector<int>::iterator start,
		   thrust::device_vector<int>::iterator end)
{
	for(; start < end; start++)
		std::cout << *start << " ";
	std::cout << std::endl;
}
*/

/*
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
*/

/*
void print_float_vec(thrust::device_vector<float>::iterator start,
		     thrust::device_vector<float>::iterator end)
{
	for(; start < end; start++)
		std::cout << *start << " ";
	std::cout << std::endl;
}
*/

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

template<typename VecType>
void print_VecArray(const viskores::cont::ArrayHandle<VecType> &arr, const viskores::Id len)
{
	auto arr_Reader = arr.ReadPortal();
	for (viskores::Id i = 0; i < arr_Reader.GetNumberOfValues(); i++)
	{
		VecType vec = arr_Reader.Get(i);
		std::cout << "( ";
		for (viskores::Id j = 0; j < len; j++)
			std::cout << vec[j] << " ";
		std::cout << ")\t";

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

/*
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
*/

struct ToRowMajor : viskores::worklet::WorkletMapField
{
	int w;
	ToRowMajor(int _w) : w(_w) {}

	using ControlSignature = void(FieldIn coordinates, FieldOut indices);
	using ExecutionSignature = _2(_1);

	VISKORES_EXEC
	int operator()(const viskores::Vec2i &pos) const
	{
		return pos[0] + pos[1] * w;
	}
};

void RasterizeTriangles(viskores::cont::ArrayHandle<viskores::Vec3f> &p1,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p2,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p3,
		viskores::cont::ArrayHandle<viskores::Vec3ui_8> &color,
		int numTri, int width, int height, Image &final_image)
{
	//Set up timing systems
	std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timer;
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
	viskores::cont::ArrayHandle<int> frags;

	FragCount fragCount;
	invoke(fragCount, p1, p2, p3, frags);

#if DEBUG > 1 
	std::cout << "# frags by triange: " << std::endl;
	print_ArrayHandle(frags);
#endif
	viskores::cont::ArrayHandle<viskores::Id> write_index;
	viskores::cont::Algorithm::ScanExclusive(viskores::cont::make_ArrayHandleCast<viskores::Id>(frags),
			write_index);
#if DEBUG > 1
	std::cout << "write position by triange: " << std::endl;
	print_ArrayHandle(write_index);
#endif

	int fragments = write_index.ReadPortal().Get(numTri-1) + frags.ReadPortal().Get(numTri-1);
#if DEBUG > 1	
	std::cout << "Number of fragments: " << fragments << std::endl;
#endif
#if DEBUG > 0
	std::cout << "Get fragments" << std::endl;
#endif

	viskores::cont::ArrayHandle<viskores::Id> frag_tri;
	vexpand(frags, frag_tri);
#if DEBUG > 3
	std::cout << "Which triangle does each fragment belong to?" << std::endl;
	print_ArrayHandle(frag_tri);
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
	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<int> rows;
	RowCount rowCount;
	invoke(rowCount, p1, p2, p3, rows);
#if DEBUG > 1
	std::cout << "How many rows does each triangle have?" << std::endl;
	print_ArrayHandle(rows);
#endif
	viskores::cont::ArrayHandle<viskores::Id> row_off;
	viskores::cont::Algorithm::ScanExclusive(viskores::cont::make_ArrayHandleCast<viskores::Id>(rows), row_off);
#if DEBUG > 1
	std::cout << "What is the row offset of each triangle?" << std::endl;
	print_ArrayHandle(row_off);
#endif

	int num_rows = row_off.ReadPortal().Get(numTri-1) + rows.ReadPortal().Get(numTri-1);

	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<viskores::Id> tri_ptr;
	
	vexpand(rows, tri_ptr);
#if DEBUG > 2 
	std::cout << "What triangle does each row belong to?" << std::endl;
	print_ArrayHandle(tri_ptr);
#endif
	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<viskores::Id> row_ptr;

	vindex(tri_ptr, row_off, row_ptr);
#if DEBUG > 2 
	std::cout << "The index of each row." << std::endl;
	print_ArrayHandle(row_ptr);
#endif
	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<int> col_count;

	ColCount colCount;
	invoke(
		colCount,
		viskores::cont::make_ArrayHandlePermutation(tri_ptr, p1),
		viskores::cont::make_ArrayHandlePermutation(tri_ptr, p2),
		viskores::cont::make_ArrayHandlePermutation(tri_ptr, p3),
		row_ptr,
		col_count
	);
#if DEBUG > 2
	std::cout << "How many columns does each row have?" << std::endl;
	print_ArrayHandle(col_count);
#endif
	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<viskores::Id> col_off;

	viskores::cont::Algorithm::ScanExclusive
		(viskores::cont::make_ArrayHandleCast<viskores::Id>(col_count), col_off);
#if DEBUG > 2 
	std::cout << "Column offsets by row" << std::endl;
	print_ArrayHandle(col_off);
	std::cout << "Number of columns " <<  col_off.ReadPortal.Get(num_rows-1) + 
		col_count.ReadPortal.Get(num_rows-1) << std::endl;
#endif
	assert((fragments == (int)col_off.ReadPortal().Get(num_rows-1) + (int)col_count.ReadPortal().Get(num_rows-1)));
	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<viskores::Id> frag_row;
	viskores::cont::ArrayHandle<viskores::Id> frag_col;

	//Determine fragment rows and columns
	vexpand(col_count, frag_row);
	std::cout << "Frag Rows" << std::endl;
	//print_ArrayHandle(frag_row);

	//temporary copies
	/*
	viskores::cont::ArrayHandle<viskores::Id> vtmp_frag_row;
	viskores::cont::ArrayHandle<viskores::Id> vtmp_frag_col;
	vtmp_frag_row.DeepCopyFrom(frag_row);
	vtmp_frag_col.DeepCopyFrom(frag_col);
	*/

	vindex(frag_row, col_off, frag_col);

	viskores::cont::Algorithm::Transform
		(frag_row,
		 viskores::cont::make_ArrayHandlePermutation(frag_tri, row_off),
		 frag_row,
		 std::minus<viskores::Id>());
	//std::cout << "Size of frag_row, frag_col: " <<
	//	frag_row.GetNumberOfValues() << ", " <<
	//	frag_col.GetNumberOfValues() << std::endl;
	//std::cout << "Frag Col" << std::endl;
	//print_ArrayHandle(frag_col);
		 
#if DEBUG > 3 
	std::cout << "Frag positions by row and column in every triangle." << std::endl;
	print_ArrayHandle(frag_row);
	print_ArrayHandle(frag_col);
#endif
	//Initialize ArrayHandles
	viskores::cont::ArrayHandle<viskores::Vec2i> pos;
	viskores::cont::ArrayHandle<float> depth;

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
		viskores::cont::make_ArrayHandlePermutation(frag_tri, p1),
		viskores::cont::make_ArrayHandlePermutation(frag_tri, p2),
		viskores::cont::make_ArrayHandlePermutation(frag_tri, p3),
		frag_row, frag_col, pos, depth);

	/*auto tmp_pos_Reader = pos.ReadPortal();
	for(viskores::Id i = 0; i < tmp_pos_Reader.GetNumberOfValues(); i++)
	{
		std::cout << "(" << thrust::get<0>(tmp_pos_Reader.Get(i)) << ", " << 
			thrust::get<1>(tmp_pos_Reader.Get(i)) << ")\t";
	}
	std::cout << std::endl;
	auto tmp_dep_Reader = depth.ReadPortal();
	for(viskores::Id i = 0; i < tmp_dep_Reader.GetNumberOfValues(); i++)
	{
		std::cout << tmp_dep_Reader.Get(i) << "\t";
	}
	std::cout << std::endl;*/

#if DEBUG > 3
	std::cout << "Position and depth of fragments" << std::endl;
	print_VecArray(pos, 2);
	print_ArrayHandle(depth);
#endif
	//Gather the color of each fragment
	viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandle<viskores::Vec3ui_8>> frag_colors(frag_tri, color);

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
	viskores::cont::ArrayHandle<viskores::Vec2i> cpos;
	cpos.DeepCopyFrom(pos);	
	viskores::cont::ArrayHandleCounting<viskores::Id> tmp_inds(0, 1, fragments);
	viskores::cont::ArrayHandle<viskores::Id> sorted_inds;
	viskores::cont::Algorithm::Copy(tmp_inds, sorted_inds);

#if DEBUG > 0
	std::cout << "\tsort fragments" << std::endl;
#endif
	viskores::cont::Algorithm::SortByKey(cpos, sorted_inds);
	viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandle<viskores::Vec3ui_8>>> cfrag_colors(sorted_inds, frag_colors);
	viskores::cont::ArrayHandlePermutation<viskores::cont::ArrayHandle<viskores::Id>, viskores::cont::ArrayHandle<float>> cdepth(sorted_inds, depth);

#if DEBUG > 3
	std::cout << "Sorted" << std::endl;
	print_VecArray(cpos, 2);
	print_ArrayHandle(sorted_inds);
	print_ArrayHandle(cdepth);
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
		viskores::cont::ArrayHandle<viskores::Vec2i> tmp_pos;
		viskores::cont::Algorithm::Copy(cpos, tmp_pos);
		viskores::cont::Algorithm::Unique(tmp_pos);
		unique_positions = tmp_pos.GetNumberOfValues();
	}
#if DEBUG > 1
	std::cout << "\tunique positions = " << unique_positions << std::endl;
#endif
*/
	viskores::cont::ArrayHandle<viskores::Vec2i> true_fragments;
	viskores::cont::ArrayHandle<float> min_depth;
	viskores::cont::ArrayHandle<int> pos_count;
	viskores::cont::Algorithm::ReduceByKey(cpos, cdepth, true_fragments, min_depth, my_maximum<float>());
	viskores::cont::Algorithm::ReduceByKey(cpos, viskores::cont::make_ArrayHandleConstant<int>(1, fragments),
		       true_fragments, pos_count, std::plus<int>());	

#if DEBUG > 3
	std::cout << "Number of duplicates at each unique position" << std::endl;
	print_ArrayHandle(pos_count);
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

	viskores::cont::ArrayHandle<float> exp_min_depth;
	vduplicate(min_depth, pos_count, exp_min_depth);

#if DEBUG > 3
	std::cout << "Min depth by fragment" << std::endl;
	print_ArrayHandle(exp_min_depth);
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
	print_ArrayHandle(exp_min_depth);
	print_ArrayHandle(cdepth);
#endif
#if DEBUG > 0
	std::cout << "\tchoose fragments to write" << std::endl;
#endif
	viskores::cont::ArrayHandle<bool> write_frag;
	viskores::cont::Algorithm::Transform(exp_min_depth, cdepth, write_frag, std::equal_to<float>());

	/*
	//Convert ArrayHandles to Thrust vectors

	//Create portals for reading
	auto pos_Reader = cpos.ReadPortal();
	auto depth_Reader = cdepth.ReadPortal();
	auto color_Reader = cfrag_colors.ReadPortal();
	auto inds_Reader = sorted_inds.ReadPortal();

	auto true_frag_reader = true_fragments.ReadPortal();
	auto min_depth_reader = min_depth.ReadPortal();
	auto pos_count_reader = pos_count.ReadPortal();
	auto exp_min_depth_reader = exp_min_depth.ReadPortal();
	auto write_frag_reader = write_frag.ReadPortal();

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
	print_ArrayHandle(write_frag);
#endif
	//time: got visible fragments
	timer.push_back(std::chrono::high_resolution_clock::now());

/*
   WRITE
*/

#if DEBUG > 0
	std::cout << "write fragments" << std::endl;
#endif

	viskores::cont::ArrayHandle<viskores::Id> rowMajorPos;
	ToRowMajor to_row_major(width);
	invoke(to_row_major, cpos, rowMajorPos);

#if DEBUG > 3
	std::cout << "Row major position by fragment" << std::endl;
	print_ArrayHandle(rowMajorPos);
#endif

	//viskores::cont::ArrayHandle<viskores:Vec3ui_8> vbg;
	//vbg.AllocateAndFill(width * height, thrust::make_tuple<char,char,char>(127,127,127));
	viskores::cont::ArrayHandle<viskores::Vec3ui_8> img;
	img.AllocateAndFill(width * height, viskores::make_Vec<viskores::UInt8>(127,127,127));
	/*
	std::cout << cfrag_colors.GetNumberOfValues() << std::endl;
	std::cout << rowMajorPos.GetNumberOfValues() << std::endl;
	std::cout << write_frag.GetNumberOfValues() << std::endl;
	std::cout << img.GetNumberOfValues() << std::endl;
	*/
	//auto max_pos = viskores::cont::Algorithm::Reduce(rowMajorPos, (viskores::Id) 0,
	//	       [](const auto& a, const auto& b){return std::max(a,b);});	
	//std::cout << max_pos << std::endl;
	FillImage<viskores::cont::StorageTagBasic> fill_image;
	invoke(
		fill_image,
		cfrag_colors,
		rowMajorPos,
		write_frag,
		img
	);

	auto img_Reader = img.ReadPortal();
	int count = 0;
	for(viskores::Id i = 0; i < img_Reader.GetNumberOfValues(); i++)
	{
		viskores::Vec3ui_8 t = img_Reader.Get(i);
		final_image.data[count++] = t[0];
		final_image.data[count++] = t[1];
		final_image.data[count++] = t[2];
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
