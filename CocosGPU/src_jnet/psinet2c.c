/*********************************************************
  psinet2c.c
  --------------------------------------------------------
  generated at Mon Aug  2 17:12:54 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

#include <math.h>

#define Act_Logistic(sum, bias)  ( (sum+bias<10000.0) ? ( 1.0/(1.0 + exp(-sum-bias) ) ) : 0.0 )
#define NULL (void *)0

typedef struct UT {
          float act;         /* Activation       */
          float Bias;        /* Bias of the Unit */
          int   NoOfSources; /* Number of predecessor units */
   struct UT   **sources; /* predecessor units */
          float *weights; /* weights from predecessor units */
        } UnitType, *pUnit;

  /* Forward Declaration for all unit types */
  static UnitType Units[81];
  /* Sources definition section */
  static pUnit Sources[] =  {
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, 
Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, 
Units + 68, Units + 69, Units + 70, Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, Units + 77, 

Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, 
Units + 68, Units + 69, Units + 70, Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, Units + 77, 

Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, 
Units + 68, Units + 69, Units + 70, Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, Units + 77, 


  };

  /* Weigths definition section */
  static float Weights[] =  {
0.313750, -0.318430, -0.051530, -0.045050, -0.155640, 0.029200, -0.199460, -0.143130, 0.399580, -0.077340, 
-0.257710, 0.415810, 0.251500, -0.281880, -0.340230, -0.118260, -0.177300, -0.181780, 0.061660, 0.014530, 
-0.143480, 0.118850, -0.014020, -0.198910, -0.056430, -0.382580, 0.144810, 0.066610, -1.095490, 1.028260, 
-0.128000, -0.479000, 0.467800, 0.240180, -0.146890, -0.098790, 0.354860, 0.019900, -0.434680, 0.390700, 
-0.224160, -0.417210, 0.168120, 0.059220, -0.572200, 0.219860, 0.064640, -0.333680, 0.201080, -0.006680, 
-0.177440, 0.248140, -0.129900, -0.026190, 0.198250, -0.209070, 0.347230, 
-0.123500, 0.589290, 0.414210, -0.182150, 0.554930, 0.467730, -0.260600, 0.604450, 0.364500, -0.389590, 
0.359660, 0.413460, -0.030200, 0.101010, 0.151220, 0.511120, -0.001310, -0.266050, 0.786270, -0.191840, 
0.049460, 0.569690, -0.347460, 0.302320, 0.361540, -0.526960, 0.486180, -0.579440, -2.100480, 2.669430, 
-0.329490, -0.884930, 1.945710, 0.087500, -0.852830, 1.587600, 0.615470, -0.758370, 1.121250, 0.933640, 
-0.482380, 0.595360, 1.026280, -0.416830, 0.375040, 1.089300, -0.066740, 0.301240, 1.038770, 0.208110, 
0.149600, 0.631580, 0.671960, -0.104630, 0.708210, 0.377660, 0.093620, 
0.280760, 0.019440, 0.124060, 0.062750, -0.240890, 0.327790, 0.437600, -0.327930, -0.373660, 0.605610, 
-0.753410, -0.115840, 1.364580, -1.124150, 0.267290, 1.750180, -1.930650, 0.545070, 1.263820, -2.287250, 
1.877730, 0.023430, -1.218320, 2.069200, -0.130170, -0.625120, 1.426670, 0.481400, -0.518820, 0.356340, 
0.994340, 0.079920, -0.505700, 1.394600, -0.029710, -0.544000, 1.450070, 0.343040, -0.820680, 1.098180, 
0.353380, -0.600690, 0.367970, 0.088350, 0.315620, 0.564570, 0.837260, -0.171780, 0.703620, 0.441190, 
-0.168950, 0.542970, 0.154600, 0.317150, 0.259860, 0.027420, 0.627410, 
0.036120, 0.452150, 0.031130, -0.176670, 0.382300, 0.178250, -0.010410, 0.445400, -0.190730, 0.156140, 
0.353800, -0.435900, 0.656410, 0.069510, -0.464430, 1.002990, -0.146340, -0.463510, 1.188060, -0.661550, 
0.164750, 0.845490, -0.998850, 0.700210, 0.350610, -0.789590, 0.606670, -0.489990, -1.373520, 1.661010, 
0.058330, -0.658540, 0.912030, 0.509540, -0.015360, 0.030060, 0.804610, 0.329220, -0.505430, 0.922420, 
0.346690, -0.595280, 0.631580, 0.094120, -0.237790, 0.454660, 0.235690, -0.103510, 0.542820, 0.203270, 
-0.003010, 0.450120, 0.309480, 0.044980, 0.190350, 0.250560, 0.313730, 
-0.335840, 0.036220, 0.320980, -0.357270, -0.220540, 0.530910, -0.139920, -0.139660, 0.352870, 0.239320, 
-0.363710, 0.074290, -0.138320, -0.301650, 0.377080, 0.204120, -0.309190, 0.349820, 0.685830, -0.579420, 
0.217190, 1.078530, -0.488260, -0.344900, 1.133430, -0.672900, -0.570300, -0.050650, -2.095310, 1.483470, 
0.860290, -0.286330, -0.576730, 0.860660, 0.691100, -1.379340, 0.616830, 0.539530, -1.091760, 0.056390, 
0.278790, -0.286570, 0.143480, 0.097610, -0.031860, 0.206840, -0.097470, 0.230620, -0.133420, -0.083950, 
0.379450, -0.246670, 0.069660, 0.397110, -0.051300, 0.205110, 0.084270, 
0.399360, -0.952670, 0.321900, -0.234710, -0.707360, -0.115710, 0.199960, -0.486460, 0.159160, -0.806570, 
-0.189320, 0.116900, 0.017150, -0.422080, -0.215640, 0.414440, -0.599410, -0.166720, 0.227670, -0.610720, 
-0.425200, -0.112960, -0.547630, -0.214970, -1.664890, -0.469080, 0.809510, -3.127020, -0.407950, 0.730640, 
-1.685090, -0.518320, -0.061640, -0.276780, -0.888360, -0.320510, 0.774940, -0.409460, -0.307490, 0.035110, 
-0.492920, -0.321890, 0.098270, -0.450650, -0.410120, 0.456510, -0.237730, -0.020270, 0.260600, 0.051510, 
0.110150, 0.010620, -0.093920, 0.037780, 0.131500, -0.386250, -0.223460, 
1.090410, -0.792170, -0.588740, -0.482160, -0.547660, 0.110500, -0.145880, 0.627790, 0.639350, 0.036330, 
0.211730, 0.271940, -0.695730, 0.680150, -0.180880, -0.148960, 0.210050, -0.519120, 0.424190, -0.725320, 
0.571590, -0.016820, -0.731240, 0.653180, 0.080870, -0.986510, 0.354510, -1.034590, -0.733620, 0.300970, 
-0.418900, -0.513440, -0.266600, 0.429630, -0.278750, -0.764040, 0.488270, 0.214710, -0.763710, 0.320560, 
0.688210, -0.880660, -0.437610, -0.079680, 0.049160, 0.301560, 0.573610, -0.090930, -0.134510, 0.244770, 
0.328290, -0.048380, 0.152870, 0.676760, -0.112290, -0.092560, 0.922680, 
-0.585270, -0.128070, 0.545190, -0.305770, -0.165620, 0.015030, -0.252750, -0.031950, 0.284810, 0.154620, 
-0.470090, -0.090800, -0.334760, -0.672820, 0.035070, -0.300600, -0.327830, 0.074820, 0.051970, 0.319570, 
-0.248310, 0.438790, 0.315470, -0.550710, 0.195050, -0.878920, 0.314560, -0.574050, -1.684230, 1.397730, 
-0.746470, -0.945590, 0.806320, 0.181880, -0.020680, -0.372280, -0.115790, -0.223990, 0.006900, -0.294640, 
-0.532200, 0.019090, -0.464180, -0.198290, -0.224450, -0.265020, 0.181630, -0.153860, -0.068080, -0.396870, 
0.283130, -0.090410, -0.205370, 0.094100, 0.204740, 0.027460, -0.575340, 
0.500890, 0.122450, -0.252570, 0.058390, -0.054400, 0.155100, 0.206440, -0.095650, 0.729040, -0.025670, 
0.175410, 0.150000, 0.235910, 0.122480, -0.400400, 0.139000, -0.204720, -0.176260, -0.214760, 0.228420, 
-0.076260, 0.621980, -0.431100, -0.328160, 1.161090, -0.568300, -1.274900, 0.137930, -1.467510, 0.558130, 
0.050570, -1.453240, 1.195380, -0.385510, -1.115840, 2.024360, -0.049250, -1.279260, 1.894470, 0.827350, 
-0.936250, 0.862870, 1.105640, -0.833280, 0.230500, 1.669580, -0.866670, 0.068970, 1.328460, -0.262310, 
0.121290, 0.642060, 0.727270, -0.536210, 1.014370, 0.379640, -0.127290, 
-0.499470, -0.127820, 0.443520, -0.274220, 0.271450, -0.020930, 0.813800, 0.135210, 0.302370, 1.506070, 
0.163630, -0.204620, 0.653340, -0.424350, -0.007890, -0.187870, -0.389150, 0.561860, -0.607500, -0.354610, 
0.779320, -0.969270, 0.225360, 0.868870, -0.597350, -0.218490, 0.158570, -1.619950, -1.373000, 2.287900, 
-1.308040, -0.985450, 1.390570, -0.884510, -0.043480, 0.442110, 0.056540, -0.367720, 0.365320, 0.268380, 
-0.336230, 0.635460, 0.327470, -0.493730, -0.093330, 0.021080, 0.230370, 0.688450, 0.663860, 0.337530, 
-0.139300, 0.731000, 0.585940, -0.336320, 0.045880, 0.083990, 0.436930, 
-0.264190, -0.008190, -0.668430, -0.400580, 0.094700, -0.726940, -0.465380, 0.129530, -0.396500, -0.131020, 
-0.380340, -0.255560, -0.058210, -1.002660, -0.008290, 0.040200, -1.030660, -0.056560, 0.422050, -1.062720, 
0.084030, 0.443470, -1.085760, 0.335900, 0.518670, -1.661570, 0.923500, 0.527590, -2.290670, 1.336090, 
1.108340, -1.648280, 0.277190, 1.118070, -1.607970, 0.183940, 0.541670, -1.323480, 0.294500, 0.357470, 
-1.101200, -0.081760, 0.318750, -0.572010, -0.342460, -0.122980, -0.205690, -0.060490, -0.217530, 0.086800, 
-0.214060, -0.106560, -0.167410, -0.246510, 0.013680, -0.211700, -0.310650, 
0.559840, -0.230290, -0.755890, 0.336740, -0.746630, -0.142630, 0.117110, -0.761720, 0.244010, 0.456190, 
-0.643280, 0.137000, 0.048340, -0.129480, 0.120440, 0.583500, -0.352900, -0.482010, 0.968610, -0.754810, 
-0.575990, 1.150360, -0.670550, -0.873990, 1.401590, -0.469420, -1.146260, 0.783590, -1.457280, 0.203670, 
0.892910, -1.005940, 0.107010, 0.722360, -0.814670, 0.199200, 0.266470, -0.375880, -0.353340, -0.377500, 
0.183460, -0.210100, -0.242880, 0.177520, -0.013700, 0.455530, -0.326960, -0.076370, 0.292370, -0.153010, 
0.132940, 0.363330, -0.368850, -0.132400, 1.120630, -1.033550, -0.139260, 
0.296620, 0.095640, -0.415180, 0.148810, -0.016230, -0.201710, 0.100030, -0.053270, -0.065000, 0.181050, 
-0.085320, -0.108350, 0.024330, 0.098770, -0.084790, 0.168570, 0.033590, -0.134750, 0.361090, -0.142440, 
-0.100390, 0.534710, -0.209860, -0.220180, 0.594370, -0.482710, -0.213700, 0.048380, -1.518890, 1.008980, 
0.594080, -0.661870, 0.107350, 0.559490, -0.103170, -0.240620, 0.385720, 0.007380, -0.309420, -0.100940, 
0.078240, 0.037130, -0.105810, -0.034470, 0.290170, 0.126270, -0.324240, 0.438860, 0.135410, -0.304360, 
0.513540, 0.089670, -0.226270, 0.379950, 0.304250, -0.301540, 0.207900, 
0.102640, 0.504580, 0.268200, 0.254000, 0.101030, 0.519490, 0.070100, -0.033360, 0.306960, -0.094860, 
-0.000660, -0.040030, 0.326840, 0.536940, -0.189320, -0.016000, 0.576250, 0.253080, -0.120130, 0.518400, 
-0.047380, -0.887400, 0.667390, -0.038370, -2.014070, 1.372070, 0.536000, -2.822430, 0.628350, 1.499690, 
-2.610210, 1.379150, 0.638510, -1.746730, 1.541300, -0.499520, -0.663280, 0.798400, -0.769180, -0.333110, 
0.526990, -0.152370, 0.118180, -0.000240, -0.124450, 0.222470, -0.495680, 0.210570, -0.022410, -0.436760, 
-0.114490, 0.089900, 0.468490, -0.329500, -0.122200, 0.766780, -0.114230, 
-0.102530, 0.480840, 0.430080, 0.212600, -0.273860, 0.813020, 0.192720, 0.394240, 0.522020, 0.590940, 
0.230890, 0.794800, 0.348840, -0.506840, 0.869830, 0.467460, 0.311750, -0.083220, -0.017490, 1.760910, 
-0.914380, 0.139990, 1.775910, -0.567720, -1.082760, 0.901540, 0.376990, -2.806570, -0.247980, 2.163030, 
-0.729650, -0.010170, -0.080040, 0.100260, -0.155400, -0.125480, 0.339210, 0.159160, 0.427910, 0.393070, 
0.197150, 0.363980, -0.051560, 0.253420, -0.313620, -0.500290, 0.764340, 0.837110, -0.370670, 0.986410, 
0.612630, 0.457990, -0.401560, 0.250630, 0.817410, -0.722340, 0.134570, 
-0.093900, 0.349360, -1.007080, -0.067860, -0.611650, -0.505090, 0.630130, -0.550920, 0.127620, 0.538730, 
-0.949430, -0.159000, -0.511020, -0.548660, -0.177150, -0.620850, -0.286760, -0.210840, -0.772230, 0.444200, 
-0.470320, 0.231980, -0.053250, -1.338840, -1.116310, -0.138690, -0.699100, -1.885660, -2.904810, 1.344830, 
-1.699590, 0.164560, -0.267610, -1.536390, 1.177320, -0.145730, -0.888060, 1.237780, -0.594640, -0.813080, 
0.545810, -0.576770, -0.217080, 0.615460, -0.937950, 1.212760, -0.706920, -0.014830, 0.269800, -0.376610, 
0.534390, 0.620900, -0.447810, 0.251080, 0.982050, -0.583260, 0.010640, 
-0.431320, 1.441920, 0.062850, 0.154300, 0.761530, -0.180160, 0.341650, 0.628670, 0.399720, 0.178800, 
0.473770, 0.223340, -0.060770, -0.141590, 0.712860, 0.073200, 0.105110, 1.133710, -0.151740, 0.504870, 
0.573890, 0.321170, 0.346950, 0.171630, -0.339180, 0.460820, 0.129510, -1.645030, 0.054230, 1.092210, 
-0.489020, -0.085640, -0.342370, -0.184990, -0.145340, -0.306940, -0.487680, 0.388800, 0.796050, -0.630260, 
0.148990, 1.070510, -0.826330, 0.220260, 0.251610, -0.109210, 0.279400, 0.203390, 0.068530, 0.333790, 
-0.049640, -0.041980, 0.569720, -0.183390, -0.370680, 0.657780, -0.012700, 
-0.429020, -0.258270, -0.703720, -0.720080, 0.143610, -0.206560, -1.135720, -0.202980, -0.415850, -1.499810, 
0.490700, -0.047610, -0.723040, 1.152860, 0.125930, -1.215920, 1.321250, -0.088920, -1.498560, 0.975900, 
0.014690, -1.894280, 0.440700, 0.158920, -1.730250, -0.580490, -0.000820, -1.321920, -2.131590, 2.208850, 
-0.502280, -0.556830, -0.095890, -0.079450, -0.512010, -0.616080, -0.124140, -0.284480, -1.362990, -1.599830, 
0.000520, -0.279670, -0.463380, -0.138830, -0.616800, -0.229040, 0.027960, 0.478750, 0.211560, -0.116650, 
-0.122950, -0.017480, -0.538610, 0.129330, -0.545410, 0.433020, -0.194330, 
0.018800, 1.047790, -0.005510, 0.351870, -0.319500, -0.800550, 0.359130, 0.000640, -0.382780, -0.232470, 
0.086760, -0.212220, 0.134490, -0.670610, 0.024490, 0.953330, -0.746910, -0.023290, 1.116080, -1.094960, 
-0.163840, 0.375220, -0.981830, 0.044780, -0.429080, -0.744940, 0.050050, -2.723850, -0.877370, 1.952820, 
0.113520, -0.583770, -1.151870, -0.284990, -1.226310, -0.186600, 0.449810, -0.790920, 0.393460, -0.070840, 
-0.470160, -0.249540, 0.135880, -0.790810, -0.006780, 0.395100, -0.592510, 0.711080, 0.007970, -0.187710, 
0.030020, -0.120720, 0.468790, -0.183290, -0.039520, 0.462360, -0.175540, 
-1.039970, -0.094750, -0.287110, -0.361110, -0.346820, -0.276220, 0.005310, 0.140960, -0.056950, 0.313130, 
-0.152070, -0.478110, -0.098890, -0.637040, 0.048050, 0.076830, -0.605150, 0.016800, -0.081260, 0.036490, 
0.021250, -0.354480, 0.049160, -0.580490, -0.773330, -0.434170, -0.993360, -0.985640, -2.160830, 0.058370, 
-1.104360, -0.185030, -0.330240, -1.627550, 1.028990, 0.082460, -2.427030, 0.889560, 0.680610, -2.331850, 
0.754890, 0.071850, -2.093050, 0.433850, -0.038110, -0.595280, -0.318440, 0.112360, -0.852000, 0.050460, 
-0.028410, -1.330660, 0.591670, -0.066350, -1.299320, 0.660000, 0.301600, 
-0.938120, 1.726450, 2.470660, 1.503340, 1.121660, -1.498560, -1.698980, -1.442030, 1.200950, -1.503070, 
0.402620, 0.731000, 0.793620, -1.872270, -1.894940, -0.443770, -0.928240, -1.230190, -1.830450, -0.231750, 

-0.472600, -1.843610, -1.556870, -0.833150, 0.695390, -0.739510, -0.847690, -1.091120, -1.588000, -0.311050, 
-2.784050, -1.120470, 0.033720, 1.131370, 3.154010, -2.638130, 2.399850, -2.754950, 0.069210, -3.219530, 

1.172840, 1.581370, 0.739610, 0.181710, -1.446830, 2.127700, 1.575810, 1.548870, 0.948410, 1.512700, 
-0.199150, -0.820620, 0.012870, -0.342560, 1.844620, 2.307660, 0.933420, 2.152940, 1.491870, 2.822650, 


  };

  /* unit definition section (see also UnitType) */
  static UnitType Units[81] = 
  {
    { 0.0, 0.0, 0, NULL , NULL },
    { /* unit 1 (Old: 1) */
      0.0, 0.001040, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 2 (Old: 2) */
      0.0, -0.003400, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 3 (Old: 3) */
      0.0, 0.001470, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 4 (Old: 4) */
      0.0, 0.000530, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 5 (Old: 5) */
      0.0, 0.001810, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 6 (Old: 6) */
      0.0, -0.003860, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 7 (Old: 7) */
      0.0, 0.004840, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 8 (Old: 8) */
      0.0, -0.000840, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 9 (Old: 9) */
      0.0, 0.003410, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 10 (Old: 10) */
      0.0, 0.001160, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 11 (Old: 11) */
      0.0, -0.001910, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 12 (Old: 12) */
      0.0, 0.004410, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 13 (Old: 13) */
      0.0, -0.004000, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 14 (Old: 14) */
      0.0, 0.003360, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 15 (Old: 15) */
      0.0, 0.003410, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 16 (Old: 16) */
      0.0, -0.000330, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 17 (Old: 17) */
      0.0, -0.001040, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 18 (Old: 18) */
      0.0, -0.000830, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 19 (Old: 19) */
      0.0, 0.002300, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 20 (Old: 20) */
      0.0, 0.001100, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 21 (Old: 21) */
      0.0, 0.002070, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 22 (Old: 22) */
      0.0, -0.002420, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 23 (Old: 23) */
      0.0, 0.002010, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 24 (Old: 24) */
      0.0, 0.004240, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 25 (Old: 25) */
      0.0, 0.000950, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 26 (Old: 26) */
      0.0, -0.003570, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 27 (Old: 27) */
      0.0, -0.004420, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 28 (Old: 28) */
      0.0, -0.001630, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 29 (Old: 29) */
      0.0, -0.003460, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 30 (Old: 30) */
      0.0, 0.004970, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 31 (Old: 31) */
      0.0, 0.000380, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 32 (Old: 32) */
      0.0, -0.000330, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 33 (Old: 33) */
      0.0, -0.002880, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 34 (Old: 34) */
      0.0, 0.002340, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 35 (Old: 35) */
      0.0, -0.004830, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 36 (Old: 36) */
      0.0, 0.001930, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 37 (Old: 37) */
      0.0, -0.001010, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 38 (Old: 38) */
      0.0, -0.001090, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 39 (Old: 39) */
      0.0, 0.002880, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 40 (Old: 40) */
      0.0, -0.003860, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 41 (Old: 41) */
      0.0, 0.001950, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 42 (Old: 42) */
      0.0, 0.003480, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 43 (Old: 43) */
      0.0, 0.001620, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 44 (Old: 44) */
      0.0, 0.001130, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 45 (Old: 45) */
      0.0, -0.001440, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 46 (Old: 46) */
      0.0, -0.002540, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 47 (Old: 47) */
      0.0, -0.001040, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 48 (Old: 48) */
      0.0, -0.000320, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 49 (Old: 49) */
      0.0, 0.000410, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 50 (Old: 50) */
      0.0, -0.001520, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 51 (Old: 51) */
      0.0, 0.001730, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 52 (Old: 52) */
      0.0, -0.004460, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 53 (Old: 53) */
      0.0, 0.001320, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 54 (Old: 54) */
      0.0, -0.004440, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 55 (Old: 55) */
      0.0, 0.004930, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 56 (Old: 56) */
      0.0, -0.000170, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 57 (Old: 57) */
      0.0, 0.002270, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 58 (Old: 58) */
      0.0, 0.486540, 57,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 59 (Old: 59) */
      0.0, 1.856540, 57,
       &Sources[57] , 
       &Weights[57] , 
      },
    { /* unit 60 (Old: 60) */
      0.0, 2.008350, 57,
       &Sources[114] , 
       &Weights[114] , 
      },
    { /* unit 61 (Old: 61) */
      0.0, 1.396490, 57,
       &Sources[171] , 
       &Weights[171] , 
      },
    { /* unit 62 (Old: 62) */
      0.0, 0.322130, 57,
       &Sources[228] , 
       &Weights[228] , 
      },
    { /* unit 63 (Old: 63) */
      0.0, -0.639140, 57,
       &Sources[285] , 
       &Weights[285] , 
      },
    { /* unit 64 (Old: 64) */
      0.0, 1.644430, 57,
       &Sources[342] , 
       &Weights[342] , 
      },
    { /* unit 65 (Old: 65) */
      0.0, 0.169060, 57,
       &Sources[399] , 
       &Weights[399] , 
      },
    { /* unit 66 (Old: 66) */
      0.0, 2.269040, 57,
       &Sources[456] , 
       &Weights[456] , 
      },
    { /* unit 67 (Old: 67) */
      0.0, 0.970850, 57,
       &Sources[513] , 
       &Weights[513] , 
      },
    { /* unit 68 (Old: 68) */
      0.0, -0.375420, 57,
       &Sources[570] , 
       &Weights[570] , 
      },
    { /* unit 69 (Old: 69) */
      0.0, 0.374120, 57,
       &Sources[627] , 
       &Weights[627] , 
      },
    { /* unit 70 (Old: 70) */
      0.0, 0.519400, 57,
       &Sources[684] , 
       &Weights[684] , 
      },
    { /* unit 71 (Old: 71) */
      0.0, -2.370250, 57,
       &Sources[741] , 
       &Weights[741] , 
      },
    { /* unit 72 (Old: 72) */
      0.0, 0.575100, 57,
       &Sources[798] , 
       &Weights[798] , 
      },
    { /* unit 73 (Old: 73) */
      0.0, -0.102890, 57,
       &Sources[855] , 
       &Weights[855] , 
      },
    { /* unit 74 (Old: 74) */
      0.0, -0.179640, 57,
       &Sources[912] , 
       &Weights[912] , 
      },
    { /* unit 75 (Old: 75) */
      0.0, -0.071520, 57,
       &Sources[969] , 
       &Weights[969] , 
      },
    { /* unit 76 (Old: 76) */
      0.0, -0.122340, 57,
       &Sources[1026] , 
       &Weights[1026] , 
      },
    { /* unit 77 (Old: 77) */
      0.0, -0.574440, 57,
       &Sources[1083] , 
       &Weights[1083] , 
      },
    { /* unit 78 (Old: 78) */
      0.0, -2.068060, 20,
       &Sources[1140] , 
       &Weights[1140] , 
      },
    { /* unit 79 (Old: 79) */
      0.0, 0.114760, 20,
       &Sources[1160] , 
       &Weights[1160] , 
      },
    { /* unit 80 (Old: 80) */
      0.0, -8.863780, 20,
       &Sources[1180] , 
       &Weights[1180] , 
      }

  };



int psinet2c(float *in, float *out, int init)
{
  int member, source;
  float sum;
  enum{OK, Error, Not_Valid};
  pUnit unit;


  /* layer definition section (names & member units) */

  static pUnit Input[57] = {Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57}; /* members */

  static pUnit Hidden1[20] = {Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, Units + 77}; /* members */

  static pUnit Output1[3] = {Units + 78, Units + 79, Units + 80}; /* members */

  static int Output[3] = {78, 79, 80};

  for(member = 0; member < 57; member++) {
    Input[member]->act = in[member];
  }

  for (member = 0; member < 20; member++) {
    unit = Hidden1[member];
    sum = 0.0;
    for (source = 0; source < unit->NoOfSources; source++) {
      sum += unit->sources[source]->act
             * unit->weights[source];
    }
    unit->act = Act_Logistic(sum, unit->Bias);
  };

  for (member = 0; member < 3; member++) {
    unit = Output1[member];
    sum = 0.0;
    for (source = 0; source < unit->NoOfSources; source++) {
      sum += unit->sources[source]->act
             * unit->weights[source];
    }
    unit->act = Act_Logistic(sum, unit->Bias);
  };

  for(member = 0; member < 3; member++) {
    out[member] = Units[Output[member]].act;
  }

  return(OK);
}
