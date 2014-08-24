#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>

using namespace std;
using namespace cv;
#define B(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3]		//B
#define G(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+1]	//G
#define R(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+2]	//R
#define ALPHA_COEFFICIENT      0.3													// Ŀ��ģ�͸���Ȩ��ȡ0.1-0.3

const int G_BIN = 8;
const int B_BIN = 8;
const int R_BIN = 8;
const int R_SHIFT = 5;
const int B_SHIFT = 5;
const int G_SHIFT = 5;

const int H_BIN = 10;
const int S_BIN = 9;

const int NParticle = 100;				// ���ӵĸ���
const int Num = 10;						//֡��ļ��
const int T = 40;						//Tf
const int Re = 30;						//
const int ai = 0.08;					//ѧϰ��
long ran_seed = 802163120; 				//���������
typedef struct SpaceState 				//�ռ�״̬
{
	int xt;               				// x����λ��
	int yt;               				// x����λ��
	float v_xt;           				// x�����˶��ٶ�
	float v_yt;           				// y�����˶��ٶ�
	int Hxt;              				// x����봰��
	int Hyt;              				// y����봰��
	float at_dot;         				// �߶ȱ任�ٶ�
}SPACESTATE;

IplImage *curFrame = NULL;				// ��ǰ֡
unsigned char * img;					//��iplimg�ĵ�char*  ���ڼ���
int Wid, Hei;							//ͼ��Ĵ�С
int WidIn, HeiIn;						//����İ��Ͱ��
int xIn, yIn;							//����ʱ��������ĵ�
int xOut, yOut;							//����ʱ��������ĵ�
int widIn,heiIn;						//����İ������
int widOut,heiOut;						//����İ������
bool bSelectObject = false;
Point origin;
Rect selection;
bool pause = false; 					//�Ƿ���ͣ
bool track = false; 					//�Ƿ����

SPACESTATE *states = NULL;	  			//״̬����
float *weights = NULL;					//ÿ�����ӵ�Ȩ��
int nbin;					    		// ֱ��ͼ����
float *modelHist = NULL;				// ���ģ�͵�ֱ��ͼ
float Pi_Thres = (float)0.90;			// Ȩ����ֵ
float Weight_Thres = (float)0.0001;		// ���Ȩ����ֵ�������ж��Ƿ�Ŀ�궪ʧ

const float DELTA_T = (float)0.05;		//֡Ƶ������Ϊ30��25��15��10��
const float VELOCITY_DISTURB = 40.0;	//�ٶ��Ŷ���ֵ
const float SCALE_DISTURB = 0.0;		//���ڿ���Ŷ�����
const float SCALE_CHANGE_D = 0.001;		//�߶ȱ任�ٶ��Ŷ���ֵ
void clearAll()
{
	if(states != NULL)
	{
		delete [] states;
	}
	if(weights != NULL)
	{
		delete [] weights;
	}
	if(modelHist != NULL)
	{
		delete [] modelHist;
	}
	delete [] img;
}

void Rgb2Hsv(float R, float G, float B, float& H, float& S, float&V)
{
     // r,g,b values are from 0 to 1
    // h = [0,360], s = [0,1], v = [0,1]
    // if s == 0, then h = -1 (undefined)
    float min, max, delta,tmp;
    tmp = R>G?G:R;
    min = tmp>B?B:tmp;
    tmp = R>G?R:G;
    max = tmp>B?tmp:B;
    V = max; // v
    delta = max - min;
    if(max != 0)
        S = delta / max; // s
    else
    {
        // r = g = b = 0 // s = 0, v is undefined
        S = 0;
        H = 0;
        return;
    }
    if (delta == 0){
        H = 0;
        return;
    }
    else if(R == max){
        if (G >= B)
            H = (G - B) / delta; // between yellow & magenta
        else
            H = (G - B) / delta + 6.0;
    }
    else if(G == max)
        H = 2.0 + (B - R) / delta; // between cyan & yellow
    else if (B == max)
        H = 4.0 + (R - G) / delta; // between magenta & cyan
    H *= 60.0; // degrees
}

void calcuColorHistogram(int x0, int y0, int Wx, int Hy,
						 unsigned char * image, int W, int H,
						 float * ColorHist, int bins)
{
	int xBegin, yBegin;//ͼ����������Ͻ�����
	int xEnd, yEnd;

	for (int i=0; i<bins; i++)
	{
		ColorHist[i] = 0.0;
	}
	if ((x0 < 0) || (x0 >= W) || (y0 < 0) || (y0 >= H)
		|| (Wx <= 0) || (Hy <= 0)) return;
	xBegin = (x0 - Wx) < 0.0 ? 0 : (x0 - Wx);
	yBegin = (y0 - Hy) < 0.0 ? 0 : (y0 - Hy);
	xEnd = (x0 + Wx) >= W ? (W - 1) : (x0 + Wx);
	yEnd = (y0 + Hy) >= H ? (H - 1) : (y0 + Hy);
	int a2 = Wx*Wx+Hy*Hy;                // ����˺����뾶ƽ��a^2
	float f = 0.0;
	int r,g,b;
	for (int j = yBegin; j<=yEnd; j++)
	{
		for (int i = xBegin; i <= xEnd; i++)
		{
			r = image[(j * W + i) * 3] >> R_SHIFT;
			g = image[(j * W + i) * 3 + 1] >> G_SHIFT;
			b = image[(j * W + i) * 3 + 2] >> B_SHIFT;
		/*	r = image[(j * W + i) * 3];
			g = image[(j * W + i) * 3 + 1];
			b = image[(j * W + i) * 3 + 2];
			float H, S, V;
			Rgb2Hsv(r, g, b, H, S, V);
			H = (int)(H / 40);
			S = (int)(S * 8);*/
			int index = r*G_BIN * B_BIN + g * B_BIN + b;
			//int index = H * S_BIN + S;
			float r2 = (float)(((j-y0)*(j-y0)+(i-x0)*(i-x0))*1.0/a2);
			float k = 1 - r2;
			f = f + k;
			ColorHist[index] =ColorHist[index] + k;
		}
	}
	for (int i = 0; i < bins; i++)
	{
		ColorHist[i] = ColorHist[i]/f;
	}
}



float randGaussian(float u, float sigma)
{
	float x1, x2, v1, v2;
	float s = 100.0;
	float y;

	/*
	ʹ��ɸѡ��������̬�ֲ�N(0,1)�������(Box-Mulles����)
	1. ����[0,1]�Ͼ����������X1,X2
	2. ����V1=2*X1-1,V2=2*X2-1,s=V1^2+V2^2
	3. ��s<=1,ת����4������ת1
	4. ����A=(-2ln(s)/s)^(1/2),y1=V1*A, y2=V2*A
	y1,y2ΪN(0,1)�������
	*/
	while (s >= 1.0 || s == 0)
	{
		x1 = (float)rand();
		x2 = (float)rand();
		x1 = x1 / RAND_MAX;
		x2 = x2 / RAND_MAX;
		v1 = 2 * x1 - 1;
		v2 = 2 * x2 - 1;
		s = v1*v1 + v2*v2;
	}
	y = (float)(sqrt(-2.0 * log(s)/s) * v1);
	/*
	���ݹ�ʽ
	z = sigma * y + u
	��y����ת����N(u,sigma)�ֲ�
	*/
	return(sigma * y + u);
}

void propagate(SPACESTATE * state, int N)
{
	int i;
	int j;
	float rn[7];

	// ��ÿһ��״̬����state[i](��N��)���и���
	for (i = 0; i < N; i++)  // �����ֵΪ0�������˹����
	{
		for (j = 0; j < 7; j++) rn[j] = randGaussian(0, (float)0.6); /* ����7�������˹�ֲ����� */
		state[i].xt = (int)(state[i].xt + state[i].v_xt * DELTA_T + rn[0] * state[i].Hxt + 0.5);
		state[i].yt = (int)(state[i].yt + state[i].v_yt * DELTA_T + rn[1] * state[i].Hyt + 0.5);
		state[i].v_xt = (float)(state[i].v_xt + rn[2] * VELOCITY_DISTURB);
		state[i].v_yt = (float)(state[i].v_yt + rn[3] * VELOCITY_DISTURB);
		state[i].Hxt = (int)(state[i].Hxt+state[i].Hxt*state[i].at_dot + rn[4] * SCALE_DISTURB + 0.5);
		state[i].Hyt = (int)(state[i].Hyt+state[i].Hyt*state[i].at_dot + rn[5] * SCALE_DISTURB + 0.5);
		state[i].at_dot = (float)(state[i].at_dot + rn[6] * SCALE_CHANGE_D);
		cvCircle(curFrame, Point(state[i].xt,state[i].yt) ,3 , CV_RGB(0,255,0),1, 8, 3);
	}
	return;
}

int initialize(int x0, int y0, int Wx, int Hy,
			   unsigned char *img, int W, int H)
{
	float rn[7];
	srand(unsigned(time(0)));
	states = new SPACESTATE [NParticle];
	assert(states != NULL);
	weights = new float [NParticle];
	assert(weights != NULL);
	nbin = R_BIN * G_BIN * B_BIN;//ȷ��ֱ��ͼ����
//	nbin = H_BIN * S_BIN;
	modelHist = new float[nbin];//����ֱ��ͼ�ڴ�
	assert(modelHist != NULL);

	//����Ŀ��ģ��ֱ��ͼ
	calcuColorHistogram(x0, y0, Wx, Hy, img, W, H, modelHist, nbin);

	//��ʼ������״̬(��(x0,y0,1,1,Wx,Hy,0.1)Ϊ���ĳ�N(0,0.4)��̬�ֲ�)
	states[0].xt = x0;
	states[0].yt = y0;
	states[0].v_xt = (float)0.0; 		// 1.0
	states[0].v_yt = (float)0.0; 		// 1.0
	states[0].Hxt = Wx;
	states[0].Hyt = Hy;
	states[0].at_dot = (float)0.0;		// 0.1
	weights[0] = (float)(1.0/NParticle);// 0.9;

	// ��ʼ������״̬(��(x0,y0,1,1,Wx,Hy,0.1)Ϊ���ĳ�N(0,0.4)��̬�ֲ�)
	for (int i = 1; i < NParticle; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			rn[j] = randGaussian(0, (float)0.6);
		}
		states[i].xt = (int)(states[0].xt + rn[0] * Wx);
		states[i].yt = (int)(states[0].yt + rn[1] * Hy);
		states[i].v_xt = (float)(states[0].v_xt + rn[2] * VELOCITY_DISTURB);
		states[i].v_yt = (float)(states[0].v_yt + rn[3] * VELOCITY_DISTURB);
		states[i].Hxt = (int)(states[0].Hxt + rn[4] * SCALE_DISTURB);
		states[i].Hyt = (int)(states[0].Hyt + rn[5] * SCALE_DISTURB);
		states[i].at_dot = (float)(states[0].at_dot + rn[6] * SCALE_CHANGE_D);
		cvCircle(curFrame, Point(states[i].xt,states[i].yt) ,3 , CV_RGB(0,255,0),1, 8, 3);
		// Ȩ��ͳһΪ1/N����ÿ����������ȵĻ���
		weights[i] = (float)(1.0/NParticle);
	}
	return 1;
}

void mouseHandler(int event, int x, int y, int flags, void *param)
{
	int centerX, centerY;
	if (bSelectObject)
	{
		selection.x = MIN(origin.x, x);
		selection.y = MIN(origin.y, y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);
	}
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		bSelectObject = true;
		track = false;
		pause = true;
		break;
	case CV_EVENT_LBUTTONUP:
		bSelectObject = false;
		centerX = selection.x + selection.width / 2;
		centerY = selection.y + selection.height / 2;
		WidIn = selection.width / 2;
		HeiIn = selection.height / 2;
		track = true;
		pause = false;
		initialize(centerX, centerY, WidIn, HeiIn, img, Wid, Hei);
		break;
	default:
		break;
	}
}


//��iplimageת��img��
void IplToImage(IplImage *src)
{
	int w = src->width;
	int h = src->height;
	for(int j = 0; j < h; j++)
	{
		for(int i = 0; i < w; i++)
		{
			img[ (j*w+i)*3 ] = R(src,i,j);
			img[ (j*w+i)*3+1 ] = G(src,i,j);
			img[ (j*w+i)*3+2 ] = B(src,i,j);
		}
	}
}

/*
�۰���ң�������NCumuWeight[N]��Ѱ��һ����С��j��ʹ��
NCumuWeight[j] <=v
float v��              һ�������������
float * NCumuWeight��  Ȩ������
int N��                ����ά��
����ֵ��
�����±����
*/
int binarySearch(float v, float * NCumuWeight, int N)
{
	int low = 0, high = N - 1;
	int n;
	while (low <= high)
	{
		n = (low + high) / 2;
		if (NCumuWeight[n] <= v && NCumuWeight[n+1] > v)
		{
			return n;
		}
		else if (NCumuWeight[n] > v)
		{
			high = n - 1;
		}
		else
		{
			low = n + 1;
		}
	}
	return 0;
}

/*
���½�����Ҫ�Բ���
���������
float * c��          ��Ӧ����Ȩ������pi(n)
int N��              Ȩ�����顢�ز�����������Ԫ�ظ���
���������
int * ResampleIndex���ز�����������
*/
void ImportanceSampling(float * c, int * ResampleIndex, int N)
{
	float rnum, *cumulateWeight;
	cumulateWeight = new float[N+1]; //�����ۼ�Ȩ�������ڴ�

	//cumulateWeight�д�ŵ���weight�е��ۼ�Ȩ�أ������ۼ�Ȩ�ؽ��й�һ��
	memset(cumulateWeight, 0.0, sizeof(cumulateWeight));
	for (int i = 0; i < N; i++)
	{
		cumulateWeight[i+1] = cumulateWeight[i] + c[i];
	}
	for (int i = 0; i < N+1; i++)
	{
		cumulateWeight[i] = cumulateWeight[i] / cumulateWeight[N];
	}

	for (int i=0; i<N; i++)
	{
		rnum = (float)rand();
		rnum = rnum / RAND_MAX;
		int j = binarySearch(rnum, cumulateWeight, N+1);
		if(j == N)
		{
			j--;
		}
		ResampleIndex[i] = j;
	}
	delete [] cumulateWeight;
	return;
}

/*
����ѡ�񣬴�N�����������и���Ȩ��������ѡ��N��
���������
SPACESTATE * state��     ԭʼ�������ϣ���N����
float * weight��         N��ԭʼ������Ӧ��Ȩ��
int N��                  ��������
���������
SPACESTATE * state��     ���¹���������
*/
void reSelect(SPACESTATE * state, float * weight, int N)
{
	SPACESTATE *tmpState;
	int i, *rsIdx;
	tmpState = new SPACESTATE[N];
	rsIdx = new int[N];

	ImportanceSampling(weight, rsIdx, N); /* ����Ȩ�����²��� */

	for (i = 0; i < N; i++)
	{
		tmpState[i] = state[rsIdx[i]];//temStateΪ��ʱ����,����state[i]��state[rsIdx[i]]������
	}
	for (i = 0; i < N; i++)
	{
		state[i] = tmpState[i];
	}

	delete[] tmpState;
	delete[] rsIdx;

	return ;
}

/*
����Bhattacharyyaϵ��
���������
float * p, * q��      ������ɫֱ��ͼ�ܶȹ���
int bins��            ֱ��ͼ����
����ֵ��
Bhattacharyyaϵ��
*/
float calcuBhattacharyya(float * p, float * q, int bins)
{
	int i;
	float rho;

	rho = 0.0;
	for (i = 0; i < bins; i++)
		rho = (float)(rho + sqrt(p[i]*q[i]));

	return(rho);
}
# define SIGMA2       0.02
float CalcuWeightedPi(float rho)
{
	float pi_n, d2;
	d2 = 1 - rho;
	pi_n = (float)(exp(- d2 / SIGMA2));
	return(pi_n);
}

/*
����weight����
���������
SPACESTATE *state��ÿ�����ӵ�״̬
float * weight: ÿ���������������ģ�����ƶ�
int N: ��������
unsigned char * image��   ͼ�����ݣ����������ң��������µ�˳��ɨ�裬
��ɫ���д���RGB, RGB, ...				 
int W, H��                ͼ��Ŀ�͸�
float * ObjectHist��      Ŀ��ֱ��ͼ
int hbins��               Ŀ��ֱ��ͼ����
���������
float * weight��          ���º��Ȩ��
����ֵ��
Bhattacharyyaϵ��
*/
void observe(SPACESTATE * state, float * weight, int N,
			 unsigned char * image, int W, int H,
			 float * ObjectHist, int hbins)
{
	float *ColorHist;
	float rho;
	ColorHist = new float[hbins];

	for (int i=0; i<N; i++)
	{
		// (1) �����ɫֱ��ͼ�ֲ�
		calcuColorHistogram(state[i].xt, state[i].yt,state[i].Hxt, state[i].Hyt,
			image, W, H, ColorHist, hbins);
		// (2) Bhattacharyyaϵ��
		rho = calcuBhattacharyya(ColorHist, ObjectHist, hbins);
		// (3) ���ݼ���õ�Bhattacharyyaϵ���������Ȩ��ֵ
		weight[i] = CalcuWeightedPi(rho);
	}
	delete [] ColorHist;
	return ;
}

/*
���ƣ�����Ȩ�أ�����һ��״̬����Ϊ�������
���������
SPACESTATE * state��      ״̬������
float * weight��          ��ӦȨ��
int N��                   ��������
���������
SPACESTATE * EstState��   ���Ƴ���״̬��
*/
void estimation(SPACESTATE * state, float * weight, int N,
				SPACESTATE & EstState)
{
	int i;
	float at_dot, Hxt, Hyt, v_xt, v_yt, xt, yt;
	float weight_sum;

	at_dot = 0;
	Hxt = 0;
	Hyt = 0;
	v_xt = 0;
	v_yt = 0;
	xt = 0;
	yt = 0;
	weight_sum = 0;
	// ���
	for (i = 0; i < N; i++)
	{
		at_dot += state[i].at_dot * weight[i];
		Hxt += state[i].Hxt * weight[i];
		Hyt += state[i].Hyt * weight[i];
		v_xt += state[i].v_xt * weight[i];
		v_yt += state[i].v_yt * weight[i];
		xt += state[i].xt * weight[i];
		yt += state[i].yt * weight[i];
		weight_sum += weight[i];
	}

	// ��ƽ��
	if (weight_sum <= 0) weight_sum = 1; // ��ֹ��0����һ�㲻�ᷢ��
	EstState.at_dot = at_dot / weight_sum;
	//EstState.Hxt = (int)(Hxt/weight_sum + 0.5);
	//EstState.Hyt = (int)(Hyt/weight_sum + 0.5);
	EstState.Hxt = (int)(Hxt / weight_sum);
	EstState.Hyt = (int)(Hyt / weight_sum);
	EstState.v_xt = v_xt / weight_sum;
	EstState.v_yt = v_yt / weight_sum;
	//EstState.xt = (int)(xt/weight_sum + 0.5);
	//EstState.yt = (int)(yt/weight_sum + 0.5);
	EstState.xt = (int)(xt / weight_sum);
	EstState.yt = (int)(yt / weight_sum);

	return;
}

/*
����ģ��
���������
SPACESTATE EstState:״̬���Ĺ���ֵ
float * TargetHist: ģ�͵���ɫֱ��ͼ
int bins:           ֱ��ͼ����
float Pit:          ģ�͸���Ȩ��
unsigned char *img: ����ͼ��
int W, H:           ͼ���͸�
�����
float *TargetHist:  ���º��ģ��ֱ��ͼ
*/
void modelUpdate(SPACESTATE EstState, float * TargetHist, int bins, float PiT,
				unsigned char * img, int W, int H)
{
	float *EstHist;	//Ŀ��ֱ��ͼ
	float Bha;		//���Ͼ���
	float Pi_E;		//���Ŷ�
	//Ԥ��Ŀ���ֱ��ͼ
	EstHist = new float[bins];
	//�ڹ���ֵ������Ŀ��ֱ��ͼ
	calcuColorHistogram(EstState.xt, EstState.yt, EstState.Hxt,
		EstState.Hyt, img, W, H, EstHist, bins);
	//����Bhattacharyyaϵ��
	Bha = calcuBhattacharyya(EstHist, TargetHist, bins);
	//�������Ȩ��
	Pi_E = CalcuWeightedPi(Bha);
	//������Ŷȴ�����ֵ
	if (Pi_E > PiT)
	{
		for (int i=0; i<bins; i++)
		{
			TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
			+ ALPHA_COEFFICIENT * EstHist[i]);
		}
	}
	delete [] EstHist;
	return;
}

/*
�ܿس������������������
���������
unsigned char * image:	��ǰ֡ͼ��
int W, int H��			ͼ��Ŀ�͸�
int &xc int &yc��		��������������ĵ�
int & Wx_h, int & Hy_h��	����������İ��Ͱ��
float & maxWeight��		������ƶ�
�����
int &xc int &yc��		��������������ĵ�
int & Wx_h, int & Hy_h��	����������İ��Ͱ��
float & maxWeight��		������ƶ�
*/
int colorParticleTracking(unsigned char * image, int W, int H
						  ,int &xc, int &yc, int & Wx_h, int & Hy_h,
						  float & maxWeight)
{
	//Ԥ��Ľ��
	SPACESTATE EState;
	// ����Ȩ���ز��������ƶȴ�ĵ������ױ�ѡ����
	reSelect(states, weights, NParticle);
	// Ϊ�ز����ĵ��������������Χ����
	propagate(states, NParticle);
	// �۲⣺����weight����
	observe(states, weights, NParticle, image, W, H,
		modelHist, nbin);
	// ���ƣ���״̬�����й��ƣ���ȡλ����
	estimation(states, weights, NParticle, EState);
	xc = EState.xt;
	yc = EState.yt;
	Wx_h = EState.Hxt;
	Hy_h = EState.Hyt;
	//����ģ��
	modelUpdate(EState, modelHist, nbin, Pi_Thres, image, W, H);
	// ����������ƶ�
	maxWeight = weights[0];
	for (int i = 1; i < NParticle; i++)
		maxWeight = maxWeight < weights[i] ? weights[i] : maxWeight;

	// ���кϷ��Լ��飬���Ϸ�����-1
	if (xc < 0 || yc < 0 || xc >= W || yc >= H ||
		Wx_h <= 0 || Hy_h <= 0)
	{
		return -1 ;
	}
	else
	{
		return 1;
	}
}

int main(int argc, char *argv[])
{
	CvCapture *capture = 0;
	capture = cvCaptureFromAVI("../13.avi");
	//capture = cvCreateCameraCapture(0);
	int row, col;
	float similarity;		//���ƶ�
	float maxWeight;
	bool start = false;

	while (capture)
	{
		curFrame = cvQueryFrame(capture);
		if (curFrame == NULL)
		{
			break;
		}

		//��ʼ��
		if (start == false)
		{
			Hei = curFrame->height;
			Wid = curFrame->width;
			img = new unsigned char [Wid * Hei * 3];
			start = true;
		}
		IplToImage(curFrame);
		maxWeight = 0.0;

		//��ʼ����
		if (track == true)
		{
			similarity = colorParticleTracking(img, Wid, Hei, xOut, yOut, widOut, heiOut, maxWeight);
			if (similarity > 0 && maxWeight > 0.0001)
			{
				cvRectangle(curFrame, cvPoint(xOut - widOut, yOut - heiOut),
					cvPoint(xOut+widOut, yOut+heiOut), cvScalar(255,0,0), 2, 8, 0);
				xIn = xOut; yIn = yOut;
				WidIn = widOut; HeiIn = heiOut;
			}
			else
			{
				cout << "target lost" << endl;
			}
		}

		cvShowImage("vedio", curFrame);
		cvSetMouseCallback("vedio", mouseHandler, 0);
		if (pause)
		{
			cvWaitKey(1000);
		}
		else
		{
			cvWaitKey(10);
		}
	}

	cvReleaseImage(&curFrame);
	clearAll();
	cvDestroyAllWindows();
}