#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	//1. 打开图片
	Mat modelImage = imread("");
	Mat image = imread("");
	if (modelImage.data == NULL) {printf("打开模板图片错误！\n"); return false}
	if (image.data == NULL) {printf("打开待处理图片错误！\n"); return false}

	namedWindow("Before", WINDOW_NORMAL);
	imshow("Before", image);

	//2. 减去背景



	//3. 显示图片

}