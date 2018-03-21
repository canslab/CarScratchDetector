#include "ConnectedComponent.h"
#include "IplImageWrapper.h"
#include <queue>

// 함수 설명 : 클래스 생성자
// 리턴 : 없음
CConnectedComponent::CConnectedComponent(void)
{
}


// 함수 설명 : 클래스 소멸자
// 리턴 : 없음
CConnectedComponent::~CConnectedComponent(void)
{
	ptArray.clear();
}


// 함수 설명 : 입력된 binary 영상으로부터 connected component(CC)들을 추출하고 각각의 CC의 정보를 vector array에 저장한다.
// 리턴 : 추출된 CC들의 정보를 담고 있는 vector array
 std::vector<CConnectedComponent> CConnectedComponent::CCFiltering( 
	 Mat binary,							// 입력 binary 영상
	 const BYTE object_color,				// binary 영상에서 CC로 추출할 object의 color 값
	 const BYTE background_color,			// binary 영상에서 CC로 취급되지 않는 background의 color 값
	 int minPixel,							// 이 값보다 적은 수의 픽셀로 이루어진 CC는 그 정보를 저장하지 않는다.
	 int maxPixel,							// 이 값보다 큰 수의 픽셀로 이루어진 CC는 그 정보를 저장하지 않는다.
	 bool longitudeTest					// 글자 성분을 판단하기 위해 가로, 세로 비율을 이용해 CC를 저장할지 아닐지를 결정하는 플래그. 1이면 비율을 테스트, 0이면 비율을 테스트하지 않음
	 )
 {
	std::vector<CConnectedComponent> ccResult;
 	Mat	ccimg;
 	binary.copyTo(ccimg);
 
 	std::queue<Point>	que;
 	std::vector<Point> vec;
 
 	int width  = binary.cols;
 	int height = binary.rows;
 
	// 8neighborhood 기준 connected component(CC) 추출
 	for( int h = 0; h < height; h++ )
 	{
 		for( int w = 0; w < width; w++ )
 		{
 			vec.clear();
 
 			if( ccimg.at<unsigned char>(h, w) == object_color )
 			{
 				int Tx, Ty, Bx, By;
 				int min_x=width, max_x=-1, min_y=height, max_y=-1;
 
 				ccimg.at<unsigned char>(h, w) = background_color;
 				que.push(Point(w, h));
 				vec.push_back(Point(w, h));
 
 				while( !que.empty() )
 				{
 					int x = que.front().x;
 					int y = que.front().y;
 
 					que.pop();
 
 					if(min_x > x)
 						min_x = x;
 					if(max_x < x)
 						max_x = x;
 					if(min_y > y)
 					{
 						min_y = y;
 						Ty = y;
 						Tx = x;
 					}
 					if(max_y < y)
 					{
 						max_y = y;
 						By = y;
 						Bx = x;
 					}			
 
 					// 8-point neighborhood
 					for(int neighx=-1; neighx<=1; neighx++)
 					{
 						for(int neighy=-1; neighy<=1; neighy++)
 						{
 							int nx = min(x+neighx, width-1);
 							nx = max(nx, 0);
 
 							int ny=min(y+neighy, height-1);
 							ny=max(ny, 0);
 
 							if(ccimg.at<unsigned char>(ny, nx) == object_color)
 							{
 								ccimg.at<unsigned char>(ny, nx) = background_color;
 								que.push(Point(nx, ny));
 								vec.push_back(Point(nx, ny));
 							}
 						}
 					}
 				}

 				int npixels = vec.size();
 
				// 글자에 해당하는 CC 추출을 위해 가로 세로 비율이 일정 범위 안에 들어오는지 테스트
 				bool bhlong;
				if( longitudeTest == true )
					bhlong = (max_y-min_y>1.2*(max_x-min_x) && max_y-min_y<9*(max_x-min_x));
				else
					bhlong = true;

				// 너무 작거나 큰 CC 제외
 				bool bSmall = (npixels < minPixel ); 
 				bool bLarge = (npixels > maxPixel );
				// bounding box 기준 object pixel의 밀도가 너무 작은 CC 제외
 				bool bDense = ((max_x-min_x)*(max_y-min_y) < 8*npixels);

 				if (!bhlong ||!bDense || bSmall || bLarge )
 				{
 					std::vector<Point>::iterator	iter;
 					for( iter = vec.begin(); iter != vec.end(); iter++ )
 					{
 						binary.at<unsigned char>((*iter).y,(*iter).x) = background_color;
 					}
 				}
 				else
 				{
					// test를 통과한 CC는 그 정보를 vector에 저장
 					CConnectedComponent ccinfo;
 					ccinfo.lu = Point(min_x, min_y);
 					ccinfo.ld = Point(min_x, max_y);
 					ccinfo.ru = Point(max_x, min_y);
 					ccinfo.rd = Point(max_x, max_y);
 					ccinfo.center = Point((double)(min_x+max_x)/2, (double)(min_y+max_y)/2);
 					ccinfo.height = max_y-min_y;
 					ccinfo.width = max_x-min_x;
 					ccinfo.T = Point(Tx, Ty);
 					ccinfo.B = Point(Bx, By);
					ccinfo.ptArray = vec;
					ccinfo.npixels = npixels;

					float avgColor = 0;
					avgColor /= vec.size();
 
 					ccResult.push_back(ccinfo);
 				}
 
 				vec.clear();
 			}
 		}
 	}
 
	return ccResult;
 }


// 함수 설명 : vector array에 존재하는 두 CC를 병합하여 하나의 CC로 만든다.
// 리턴 : 없음
void CConnectedComponent::mergeCC(
	std::vector<CConnectedComponent> &ccList,		// 병합을 수행할 CC들을 담고 있는 vector array
	int index1,										// 병합을 수행할 CC1의 vector array상에서의 index
	int index2										// 병합을 수행할 CC2의 vector array상에서의 index
	)
{
	if( index1 < 0 || index2 < 0 || index1 > ccList.size() || index2 > ccList.size() )
	{
		printf("wrong merge in connected component handling!\n");
		return;
	}

	int min_x = ccList[index1].lu.x;
	int min_y = ccList[index1].lu.y;
	int max_x = ccList[index1].ru.x;
	int max_y = ccList[index1].rd.y;
	Point T = ccList[index1].T;
	Point B = ccList[index1].B;

	for( size_t i = 0; i < ccList[index2].ptArray.size(); i++ )
	{
		Point currentPoint = ccList[index2].ptArray[i];
		int x = currentPoint.x;
		int y = currentPoint.y;
		if(min_x > x)
			min_x = x;
		if(max_x < x)
			max_x = x;
		if(min_y > y)
		{
			min_y = y;
			T = currentPoint;
		}
		if(max_y < y)
		{
			max_y = y;
			B = currentPoint;
		}
		ccList[index1].ptArray.push_back(currentPoint);
	}

	ccList[index1].lu = Point(min_x, min_y);
	ccList[index1].ld = Point(min_x, max_y);
	ccList[index1].ru = Point(max_x, min_y);
	ccList[index1].rd = Point(max_x, max_y);
	ccList[index1].center = Point((double)(min_x+max_x)/2, (double)(min_y+max_y)/2);
	ccList[index1].height = max_y-min_y;
	ccList[index1].width = max_x-min_x;
	ccList[index1].T = T;
	ccList[index1].B = B;

	ccList.erase(ccList.begin()+index2);

}