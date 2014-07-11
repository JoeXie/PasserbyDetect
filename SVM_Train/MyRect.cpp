#include "MyRect.h"


MyRect::MyRect(int x, int y, int width, int height): cv::Rect(x, y, width, height)
{}

MyRect::MyRect(cv::Rect & rect): cv::Rect(rect.x, rect.y, rect.width, rect.height)
{}

int MyRect::getAera()
{
	return this->width * this->height;
}

int MyRect::getOverlapAera(MyRect rectangle)
{
	MyRect overlap = *this & rectangle;
	return overlap.getAera();
}

void MyRect::draw(IplImage* img)
{
	cvRectangle(img, this->tl(), this->br(), cv::Scalar(0, 255, 0), 2);
}