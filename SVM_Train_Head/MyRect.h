#include <opencv2/opencv.hpp>

class MyRect : public cv::Rect
{
public:
	MyRect(int x, int y, int width, int height);
	MyRect(cv::Rect & rect);

	int getX()
	{
		return this->x;
	}

	int getY()
	{
		return this->y;
	}

	int getWidth()
	{
		return this->width;
	}

	int getHeight()
	{
		return this->height;
	}

	int getAera();
	int getOverlapAera(MyRect rectangle);
	void draw(IplImage* img);

};

