#include "ConnectedComponent.h"
#include "IplImageWrapper.h"
#include <queue>

// �Լ� ���� : Ŭ���� ������
// ���� : ����
CConnectedComponent::CConnectedComponent(void)
{
}


// �Լ� ���� : Ŭ���� �Ҹ���
// ���� : ����
CConnectedComponent::~CConnectedComponent(void)
{
	ptArray.clear();
}


// �Լ� ���� : �Էµ� binary �������κ��� connected component(CC)���� �����ϰ� ������ CC�� ������ vector array�� �����Ѵ�.
// ���� : ����� CC���� ������ ��� �ִ� vector array
 std::vector<CConnectedComponent> CConnectedComponent::CCFiltering( 
	 Mat binary,							// �Է� binary ����
	 const BYTE object_color,				// binary ���󿡼� CC�� ������ object�� color ��
	 const BYTE background_color,			// binary ���󿡼� CC�� ��޵��� �ʴ� background�� color ��
	 int minPixel,							// �� ������ ���� ���� �ȼ��� �̷���� CC�� �� ������ �������� �ʴ´�.
	 int maxPixel,							// �� ������ ū ���� �ȼ��� �̷���� CC�� �� ������ �������� �ʴ´�.
	 bool longitudeTest					// ���� ������ �Ǵ��ϱ� ���� ����, ���� ������ �̿��� CC�� �������� �ƴ����� �����ϴ� �÷���. 1�̸� ������ �׽�Ʈ, 0�̸� ������ �׽�Ʈ���� ����
	 )
 {
	std::vector<CConnectedComponent> ccResult;
 	Mat	ccimg;
 	binary.copyTo(ccimg);
 
 	std::queue<Point>	que;
 	std::vector<Point> vec;
 
 	int width  = binary.cols;
 	int height = binary.rows;
 
	// 8neighborhood ���� connected component(CC) ����
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
 
				// ���ڿ� �ش��ϴ� CC ������ ���� ���� ���� ������ ���� ���� �ȿ� �������� �׽�Ʈ
 				bool bhlong;
				if( longitudeTest == true )
					bhlong = (max_y-min_y>1.2*(max_x-min_x) && max_y-min_y<9*(max_x-min_x));
				else
					bhlong = true;

				// �ʹ� �۰ų� ū CC ����
 				bool bSmall = (npixels < minPixel ); 
 				bool bLarge = (npixels > maxPixel );
				// bounding box ���� object pixel�� �е��� �ʹ� ���� CC ����
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
					// test�� ����� CC�� �� ������ vector�� ����
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


// �Լ� ���� : vector array�� �����ϴ� �� CC�� �����Ͽ� �ϳ��� CC�� �����.
// ���� : ����
void CConnectedComponent::mergeCC(
	std::vector<CConnectedComponent> &ccList,		// ������ ������ CC���� ��� �ִ� vector array
	int index1,										// ������ ������ CC1�� vector array�󿡼��� index
	int index2										// ������ ������ CC2�� vector array�󿡼��� index
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