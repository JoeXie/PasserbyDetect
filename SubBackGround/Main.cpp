#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	//1. ��ͼƬ
	Mat modelImage = imread("");
	Mat image = imread("");
	if (modelImage.data == NULL) {printf("��ģ��ͼƬ����\n"); return false}
	if (image.data == NULL) {printf("�򿪴�����ͼƬ����\n"); return false}

	namedWindow("Before", WINDOW_NORMAL);
	imshow("Before", image);

	//2. ��ȥ����



	//3. ��ʾͼƬ

}