#ifndef SDOPENCV_ITERATORS_HPP
#define SDOPENCV_ITERATORS_HPP

#include <cxcore.h>
#include <cv.h>
#include <cstring>

namespace sdopencv
{

inline cv::MemStorage createMemStorage(int block_size CV_DEFAULT(0))
{
    return cv::MemStorage(cvCreateMemStorage(block_size));
}

inline cv::MemStorage createChildMemStorage(cv::MemStorage &parent)
{
    return cv::MemStorage(cvCreateChildMemStorage((CvMemStorage *)parent));
}

class CV_EXPORTS LineIterator : public cv::LineIterator
{
public:
    LineIterator(const cv::Mat& img, cv::Point const &pt1, cv::Point const &pt2,
        int connectivity=8, bool leftToRight=false);
        
    LineIterator const &iter() { return *this; }
    cv::Point next();
    
private:
    int iteration;
    int ws, es;
    uchar *ptr0;
};

template<typename T>
struct unbounded_array
{
    T &operator[](unsigned int i) { return ((T *)(this))[i]; }
};

}

namespace cv // missing classes in OpenCV 2.1
{
typedef Size_<double> Size2d;
typedef Rect_<float> Rectf;
typedef Rect_<double> Rectd;

struct CV_EXPORTS VectorBase {};


//////////////////////////////// SdVector ////////////////////////////////

// template vector class. It is similar to STL's vector,
// with a few important differences:
//   1) it can be created on top of user-allocated data w/o copying it
//   2) vector b = a means copying the header,
//      not the underlying data (use clone() to make a deep copy)
template <typename _Tp> class CV_EXPORTS SdVector : public VectorBase
{
public:
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef _Tp& reference;
    typedef const _Tp& const_reference;
    
    struct CV_EXPORTS Hdr
    {
        Hdr() : data(0), datastart(0), refcount(0), size(0), capacity(0) {};
        _Tp* data;
        _Tp* datastart;
        int* refcount;
        size_t size;
        size_t capacity;
    };
    
    SdVector() {}
    SdVector(size_t _size)  { resize(_size); }
    SdVector(size_t _size, const _Tp& val)
    {
        resize(_size);
        for(size_t i = 0; i < _size; i++)
            hdr.data[i] = val;
    }
    SdVector(_Tp* _data, size_t _size, bool _copyData=false)
    { set(_data, _size, _copyData); }
    
    template<int n> SdVector(const Vec<_Tp, n>& vec)
    { set((_Tp*)&vec.val[0], n, true); }    
    
    SdVector(const std::vector<_Tp>& vec, bool _copyData=false)
    { set((_Tp*)&vec[0], vec.size(), _copyData); }    
    
    SdVector(const SdVector& d) { *this = d; }
    
    SdVector(const SdVector& d, Range r)
    {
        if( r == Range::all() )
            r = Range(0, d.size());
        if( r.size() > 0 && r.start >= 0 && r.end <= d.size() )
        {
            if( d.hdr.refcount )
                CV_XADD(d.hdr.refcount, 1);
            hdr.refcount = d.hdr.refcount;
            hdr.datastart = d.hdr.datastart;
            hdr.data = d.hdr.data + r.start;
            hdr.capacity = hdr.size = r.size();
        }
    }
    
    SdVector<_Tp>& operator = (const SdVector& d)
    {
        if( this != &d )
        {
            if( d.hdr.refcount )
                CV_XADD(d.hdr.refcount, 1);
            release();
            hdr = d.hdr;
        }
        return *this;
    }
    
    ~SdVector()  { release(); }
    
    SdVector<_Tp> clone() const
    { return hdr.data ? SdVector<_Tp>(hdr.data, hdr.size, true) : SdVector<_Tp>(); }
    
    void copyTo(SdVector<_Tp>& vec) const
    {
        size_t i, sz = size();
        vec.resize(sz);
        const _Tp* src = hdr.data;
        _Tp* dst = vec.hdr.data;
        for( i = 0; i < sz; i++ )
            dst[i] = src[i];
    }
    
    void copyTo(std::vector<_Tp>& vec) const
    {
        size_t i, sz = size();
        vec.resize(sz);
        const _Tp* src = hdr.data;
        _Tp* dst = sz ? &vec[0] : 0;
        for( i = 0; i < sz; i++ )
            dst[i] = src[i];
    }
    
    _Tp& operator [] (size_t i) { CV_Assert( i < size() ); return hdr.data[i]; }
    const _Tp& operator [] (size_t i) const { CV_Assert( i < size() ); return hdr.data[i]; }
    SdVector operator() (const Range& r) const { return SdVector(*this, r); }
    _Tp& back() { CV_Assert(!empty()); return hdr.data[hdr.size-1]; }
    const _Tp& back() const { CV_Assert(!empty()); return hdr.data[hdr.size-1]; }
    _Tp& front() { CV_Assert(!empty()); return hdr.data[0]; }
    const _Tp& front() const { CV_Assert(!empty()); return hdr.data[0]; }
    
    _Tp* begin() { return hdr.data; }
    _Tp* end() { return hdr.data + hdr.size; }
    const _Tp* begin() const { return hdr.data; }
    const _Tp* end() const { return hdr.data + hdr.size; }
    
    void addref() { if( hdr.refcount ) CV_XADD(hdr.refcount, 1); }
    void release()
    {
        if( hdr.refcount && CV_XADD(hdr.refcount, -1) == 1 )
        {
            delete[] hdr.datastart;
            delete hdr.refcount;
        }
        hdr = Hdr();
    }
    
    void set(_Tp* _data, size_t _size, bool _copyData=false)
    {
        if( !_copyData )
        {
            release();
            hdr.data = hdr.datastart = _data;
            hdr.size = hdr.capacity = _size;
            hdr.refcount = 0;
        }
        else
        {
            reserve(_size);
            for( size_t i = 0; i < _size; i++ )
                hdr.data[i] = _data[i];
            hdr.size = _size;
        }
    }
    
    void reserve(size_t newCapacity)
    {
        _Tp* newData;
        int* newRefcount;
        size_t i, oldSize = hdr.size;
        if( (!hdr.refcount || *hdr.refcount == 1) && hdr.capacity >= newCapacity )
            return;
        newCapacity = std::max(newCapacity, oldSize);
        newData = new _Tp[newCapacity];
        newRefcount = new int(1);
        for( i = 0; i < oldSize; i++ )
            newData[i] = hdr.data[i];
        release();
        hdr.data = hdr.datastart = newData;
        hdr.capacity = newCapacity;
        hdr.size = oldSize;
        hdr.refcount = newRefcount;
    }
    
    void resize(size_t newSize)
    {
        size_t i;
        newSize = std::max(newSize, (size_t)0);
        if( (!hdr.refcount || *hdr.refcount == 1) && hdr.size == newSize )
            return;
        if( newSize > hdr.capacity )
            reserve(std::max(newSize, std::max((size_t)4, hdr.capacity*2)));
        for( i = hdr.size; i < newSize; i++ )
            hdr.data[i] = _Tp();
        hdr.size = newSize;
    }
    
    SdVector<_Tp>& push_back(const _Tp& elem)
    {
        if( hdr.size == hdr.capacity )
            reserve( std::max((size_t)4, hdr.capacity*2) );
        hdr.data[hdr.size++] = elem;
        return *this;
    }
    
    SdVector<_Tp>& pop_back()
    {
        if( hdr.size > 0 )
            --hdr.size;
        return *this;
    }
    
    size_t size() const { return hdr.size; }
    size_t capacity() const { return hdr.capacity; }
    bool empty() const { return hdr.size == 0; }
    void clear() { resize(0); }
    int type() const { return DataType<_Tp>::type; }
    
    // Minh-Tri
    void setitem(size_t i, _Tp const &value) { CV_Assert( i < size() ); hdr.data[i] = value; }
    operator cv::Mat() const { return cv::Mat(1, (int)size(), type(), (void*)hdr.data); }
    
    
protected:
    Hdr hdr;
};    

    
template<typename _Tp> inline typename DataType<_Tp>::work_type
dot(const SdVector<_Tp>& v1, const SdVector<_Tp>& v2)
{
    typedef typename DataType<_Tp>::work_type _Tw;
    size_t i, n = v1.size();
    assert(v1.size() == v2.size());

    _Tw s = 0;
    const _Tp *ptr1 = &v1[0], *ptr2 = &v2[0];
    for( i = 0; i <= n - 4; i += 4 )
        s += (_Tw)ptr1[i]*ptr2[i] + (_Tw)ptr1[i+1]*ptr2[i+1] +
            (_Tw)ptr1[i+2]*ptr2[i+2] + (_Tw)ptr1[i+3]*ptr2[i+3];
    for( ; i < n; i++ )
        s += (_Tw)ptr1[i]*ptr2[i];
    return s;
}

}

template<typename T> inline int _cmp(T const &inst1, T const &inst2)
    { return std::strncmp((char const *)&inst1, (char const *)&inst2, sizeof(T)); }

#define DEFINE_CMP_OPERATORS(T) \
inline bool operator<(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)<0; } \
inline bool operator<=(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)<=0; } \
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; } \
inline bool operator!=(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)!=0; } \
inline bool operator>=(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)>=0; } \
inline bool operator>(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)>0; }

#define DEFINE_EQUAL_OPERATOR(T) \
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; }

namespace cv {

template<typename T>
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; }

}

template<typename T>
inline bool operator==(T const &inst1, T const &inst2) { return _cmp<T>(inst1, inst2)==0; }


#endif // SDOPENCV_ITERATORS_HPP
