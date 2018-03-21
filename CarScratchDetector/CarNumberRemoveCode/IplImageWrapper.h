#pragma once
#include "opencv2/opencv.hpp"

template<class T> 
class CIplimageWrapper
{
private:
	// IplImage* imgp;

public:
	CIplimageWrapper(IplImage* img=0) : _row(NULL)
	{
		if( img == NULL )
			return;
	//	imgp=img;
		_row = new T*[img->height];
		
		_row[0] = (T*)img->imageData;
		for( int i=1; i<img->height; i++ )
			_row[i] = (T*)( (BYTE*)_row[i-1] + img->widthStep );

	}
	void	Bind(IplImage* img) 
	{
		//	imgp=img;
		_row = new T*[img->height];

		_row[0] = (T*)img->imageData;
		for( int i=1; i<img->height; i++ )
			_row[i] = (T*)( (BYTE*)_row[i-1] + img->widthStep );
	}

	~CIplimageWrapper()
	{
	//	imgp=0;
		if( _row )
			delete [] _row;
	}

	//void	Bind(IplImage* img)		{imgp=img;}
	//void operator=(IplImage* img)	{imgp=img;}
	
	inline T* operator[](const int rowIndx) 
	{
		return _row[rowIndx];
	}

	inline T* operator[](const int rowIndx)  const
	{
		return _row[rowIndx];
	}

protected:
	T**	_row;
};

typedef struct{
	unsigned char b,g,r;
} RgbPixel;

typedef struct{
	float b,g,r;
} RgbPixelFloat;

typedef	unsigned char	BYTE;

typedef CIplimageWrapper<RgbPixel>			RgbImage;
typedef CIplimageWrapper<RgbPixelFloat>	RgbImageFloat;
typedef CIplimageWrapper<BYTE>				BwImage;
typedef CIplimageWrapper<float>			BwImageFloat;
typedef CIplimageWrapper<int>			BwImageInt;