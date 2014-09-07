/*********************************************************
  hmm2.c
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:19 1999
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
-0.051950, 0.421710, -0.461510, 0.101230, 0.034530, -0.224230, -0.043870, -0.108480, 0.076350, 0.218560, 
-0.071830, 0.064020, 0.152070, -0.317110, 0.107100, 0.168720, -0.572270, 0.189530, 0.600510, -0.496410, 
0.177670, 0.444420, -0.630490, 0.219060, 0.211510, -0.401000, 0.432890, -0.001090, -0.090730, 0.795980, 
0.148670, -0.264050, 0.677050, 0.397410, -0.384330, 0.501700, 0.754590, -0.443950, 0.411740, 1.004490, 
-0.357050, -0.064220, 0.763850, -0.308170, 0.195530, 0.116370, 0.201150, 0.324390, 0.009390, 0.223640, 
-0.027900, -0.114910, 0.435300, -0.161720, -0.107310, 0.309570, -0.259460, 
-0.640420, 0.587790, 0.545770, -0.393940, 0.402130, 0.444210, -0.498540, 0.635430, 0.086380, -0.154130, 
0.112270, 0.116020, -0.019410, 0.160200, 0.014160, 0.272090, 0.164990, 0.066260, 0.404300, 0.097350, 
0.155510, 0.759610, -0.047540, 0.099730, 0.666950, -0.727650, 0.377440, -0.373600, -1.943110, 2.435380, 
-0.566340, -1.136010, 2.037990, -0.230690, -0.901120, 1.732270, 0.262660, -0.490400, 1.162100, 0.947630, 
-0.157260, 0.445620, 1.206180, -0.043470, -0.173660, 0.902450, 0.250640, -0.343050, 0.741920, 0.418210, 
-0.137060, 0.483110, 0.379580, 0.039790, 0.415370, -0.085310, 0.407160, 
-0.314630, -0.183900, 0.858060, -0.072050, -0.177400, 0.606180, 0.334170, -0.369040, 0.469000, 0.585510, 
-0.446790, 0.325820, 0.703020, -0.943570, 0.861110, 0.524350, -1.462170, 1.676410, 0.261110, -2.062980, 
2.123300, -0.220190, -1.413140, 1.819870, -0.463040, -0.662790, 1.192120, -0.438030, -0.313210, 0.887320, 
0.331820, -0.087760, 0.181120, 0.799800, -0.056020, -0.146510, 0.484610, 0.170370, -0.114480, 0.383250, 
-0.001130, 0.212720, -0.037710, 0.189820, 0.373680, 0.044250, -0.045210, 0.657020, 0.284750, -0.216670, 
0.620040, 0.183470, 0.167290, 0.366110, 0.057370, 0.700760, 0.113220, 
-0.251110, 0.404960, 0.392860, -0.198070, 0.459940, 0.189330, 0.055050, 0.283540, 0.191160, 0.326470, 
0.074640, 0.098350, 0.448020, -0.140550, 0.235240, 0.394120, -0.306310, 0.546680, 0.322770, -0.785710, 
0.988070, 0.132780, -0.766180, 1.085310, -0.191170, -0.580320, 1.072330, -0.389530, -0.974990, 1.664440, 
0.042680, -0.554840, 1.041030, 0.532610, -0.041340, 0.241220, 0.654360, 0.459870, -0.322290, 0.663030, 
0.563240, -0.401980, 0.245920, 0.642380, -0.171700, 0.150530, 0.480320, 0.076610, 0.213780, 0.314680, 
0.253580, 0.195720, 0.180520, 0.396590, 0.124840, 0.156900, 0.606310, 
-0.612300, 0.232360, -0.153140, -0.418740, 0.087100, -0.186890, -0.163680, -0.086860, -0.320070, 0.011230, 
-0.400730, -0.376460, -0.318370, -0.639210, 0.066360, -0.480680, -0.359270, 0.122220, -0.703040, -0.344410, 
-0.200840, -0.189730, -0.403260, -0.353760, -0.036390, -0.476940, -0.554020, 0.209230, -1.473820, 0.310840, 
-0.362360, -0.321430, 0.131440, -1.130430, 0.176730, 0.613870, -1.683180, 0.433890, 0.765040, -1.524590, 
0.467930, 0.808180, -0.960100, 0.345800, 0.172760, -0.200940, 0.072430, -0.443590, 0.034400, 0.304140, 
-0.402360, -0.051240, 0.306930, -0.235220, -0.101610, 0.276850, -0.296630, 
-0.846560, 0.056740, 0.239490, -0.209040, -0.384060, 0.042580, -0.107680, -0.447890, 0.087370, 0.078780, 
-0.760950, 0.183960, 0.105880, -0.874720, 0.283970, -0.367520, -0.807290, 0.194390, 0.066890, -0.862880, 
0.146740, 0.289330, -0.539220, -0.170170, -0.363250, -0.274170, -0.496390, -2.617880, -0.967470, 1.249290, 
-0.937350, 0.641260, -0.107910, -0.470980, 0.520790, -0.269940, -0.182070, -0.300200, 0.307940, -0.358080, 
0.326800, 0.178360, -0.149470, -0.272590, 0.132220, -0.867960, 0.350350, -0.271200, -0.814580, -0.017980, 
0.064270, -0.561590, -0.185170, -0.292370, -0.024670, 0.308980, -0.779140, 
0.330710, 0.135330, -0.846790, 0.140970, -0.047760, -0.221280, 0.356870, 0.007430, -0.112380, 0.047440, 
-0.002210, 0.007260, 0.319400, 0.325210, -0.069660, 0.695420, -0.320040, -0.126350, 0.776130, -1.295490, 
0.695020, -0.010760, -1.586560, 1.055400, -0.807710, -0.748270, 1.076780, -1.118910, 0.664480, 0.015260, 
0.174030, 0.475800, -0.348360, 0.673720, 0.304530, -0.434070, 1.051840, 0.345980, -0.889690, 0.697900, 
-0.045750, 0.063860, 0.227970, -0.075770, 0.208640, 0.301360, 0.216990, 0.169890, 0.301500, -0.170580, 
0.258830, 0.126730, 0.062860, 0.053580, -0.498630, 0.879110, 0.279090, 
-0.822040, 0.044170, 0.513960, -0.124670, -0.198970, 0.261350, 0.035220, -0.514610, 0.618130, 0.132630, 
-1.107650, 0.705780, 0.278200, -1.213540, 0.525000, 0.177840, -1.336010, 0.115640, 0.591980, -1.204280, 
-0.075960, 0.634270, -0.521820, -0.680300, -0.047370, -0.220840, -0.471040, -1.599370, -1.083350, 0.901870, 
-0.852640, -0.805790, 0.901140, -0.406420, -0.743310, 0.656400, 0.222860, -0.864150, -0.039200, 0.490570, 
-0.880610, -0.291550, 0.539390, -0.773320, -0.475990, -0.127660, -0.813310, 0.028680, -0.646910, -0.475010, 
0.365680, -0.013970, -0.336390, -0.410330, 0.545800, 0.083380, -0.910800, 
-0.082300, 0.166300, -0.200010, 0.179730, -0.193390, -0.111280, 0.090740, 0.021320, -0.392730, 0.043580, 
-0.309390, -0.257720, -0.152290, -0.090840, -0.109840, -0.438370, 0.566780, 0.138810, -0.251610, 0.497860, 
0.001250, 0.749420, 0.155830, -0.206380, 0.965750, -0.357010, -0.739580, 0.135710, -1.672050, 0.751000, 
-0.253280, 0.116600, 0.316060, -0.796980, -0.256710, 1.182240, -0.566150, -0.761940, 1.675280, 0.390710, 
-0.588190, 1.187160, 0.509620, -0.440820, 0.329260, 0.124510, -0.143060, -0.089750, 0.059220, 0.466330, 
0.114740, 0.091460, 0.538840, -0.118750, -0.006370, -0.209510, 0.609670, 
-0.101120, -0.387610, 0.169170, -0.660290, -0.415430, 0.257800, -0.055430, -0.222790, -0.150790, -0.423630, 
-0.153360, 0.519810, 0.603720, 0.102860, 0.237960, 0.427750, 0.502240, -0.279420, 0.579820, 0.616090, 
0.047970, 0.206740, -0.138610, 0.562840, -0.554900, -0.301480, 1.436660, -2.353540, -0.642150, 3.672320, 
-0.631980, -0.544700, 1.413160, -0.201440, -0.149190, 0.358560, 0.382310, 0.857840, -0.321570, 0.845330, 
0.986590, -0.010630, 0.347620, 0.584230, 0.034150, -0.112660, 0.079060, 1.022340, 0.254720, -0.210850, 
0.564710, 0.412120, -0.095860, -0.102160, 0.564090, 0.017080, 0.085990, 
0.299600, 0.564330, -0.185630, -0.208650, 0.565920, 0.103930, -0.383980, 0.608710, -0.084100, -0.069070, 
0.548360, -0.654780, 0.484610, 0.016880, -0.656620, 0.884600, -0.317800, -0.419680, 1.099910, -0.818190, 
-0.254370, 1.112320, -1.049330, 0.203060, -0.043400, -0.863300, 0.822870, -0.945750, -1.152500, 1.410400, 
-0.646460, 0.016790, 0.349740, 0.510800, 0.447910, -0.747020, 0.588790, 0.335770, -0.823710, 0.522870, 
0.020880, -0.455940, 0.171240, 0.057120, 0.023350, 0.105800, -0.385160, 0.572150, 0.432600, -0.259220, 
0.385760, 0.479910, 0.145130, 0.037520, 0.395960, 0.060750, 0.185870, 
-0.017180, 0.136100, -0.119030, -0.267780, 0.164390, -0.139020, -0.284970, 0.082740, -0.091540, 0.045550, 
-0.583660, 0.027090, 0.339690, -1.101850, 0.113190, 0.729390, -1.295280, -0.205160, 0.921140, -1.165030, 
-0.683780, 1.235400, -0.849140, -0.857490, 0.950190, -0.598540, -0.621780, 0.717630, -1.364290, 0.233180, 
0.292860, -1.112480, 0.718000, -0.145220, -0.785680, 0.795430, -0.606810, -0.259870, 0.518250, -0.504880, 
0.076390, 0.235790, -0.148020, 0.008570, -0.228370, -0.108100, 0.125860, -0.454330, -0.117590, 0.189340, 
-0.145190, -0.112550, -0.019140, 0.157510, -0.142170, -0.082530, 0.331640, 
-0.213710, 0.158760, -0.420740, -0.224590, 0.222480, -0.588820, -0.184600, 0.267230, -0.718120, -0.184130, 
-0.084370, -0.499300, -0.189470, -0.327130, -0.217730, -0.219430, -0.605590, -0.057240, -0.069870, -0.909930, 
0.067740, 0.244690, -1.114830, 0.226690, 0.384090, -1.158530, 0.302130, 0.729280, -1.566240, 0.318080, 
0.829270, -1.287740, 0.026840, 0.579070, -1.033140, -0.118680, 0.207070, -0.802640, -0.136740, 0.030900, 
-0.593390, -0.036780, -0.048510, -0.442940, -0.098800, -0.231400, -0.342120, 0.023050, -0.466020, -0.065610, 
0.142830, -0.389450, -0.056450, 0.089640, -0.235390, -0.075030, -0.013660, 
-0.490080, 0.255900, -0.259210, -0.163630, 0.086090, -0.029920, 0.203390, 0.154710, -0.361910, 0.175110, 
0.262120, -0.040600, 0.025890, 0.364630, 0.026840, 0.078230, 0.175640, -0.007610, 0.294770, 0.755190, 
0.289290, 0.043630, 0.885240, 0.297500, -0.844800, 1.354290, 1.237560, -1.554700, 1.430320, 2.098470, 
-1.731370, 1.524330, 0.711360, -0.857020, 1.150810, -0.271540, -0.377700, 0.297640, 0.012480, -0.591570, 
0.092790, 0.046760, -0.420400, 0.028440, -0.076780, -0.215550, 0.057310, -0.159800, -0.161840, -0.600930, 
-0.439110, -0.261930, -0.125560, -0.652880, -1.188840, 0.196340, -0.552970, 
-0.353140, 0.579900, -0.306070, -0.271210, 0.706120, 0.184390, 0.209610, 0.842320, 0.149780, 0.431850, 
0.231890, 0.328740, 0.268870, 0.514450, 0.436110, -0.283620, 0.154240, 0.256670, -0.691340, 0.368400, 
0.572810, -0.213150, 0.431950, -0.209240, -0.757650, 0.851850, -0.206490, -1.190390, -0.355240, 0.677500, 
-0.869550, 1.398940, -0.087710, -0.468030, 0.811150, 0.240940, -0.382550, 0.722940, 0.655710, -0.343710, 
0.346990, 1.480150, 0.132870, 0.844680, 0.523690, 0.681100, 0.886740, 0.031050, 0.273600, 1.011210, 
-0.182380, -0.297000, 0.739070, 0.026900, -1.061770, 0.564150, 0.534960, 
0.644200, -0.108250, -0.837180, 0.219170, -0.711760, 0.307360, 0.015140, -0.555930, 0.232940, 0.405210, 
-0.435270, -0.631290, 0.393410, -0.304730, -1.252690, -0.176530, -0.165130, -1.135450, -0.357480, -0.341180, 
-0.604000, -0.384930, -0.273860, -0.366970, -0.864250, -0.161710, -0.461520, -1.993320, -0.637910, 0.761480, 
-1.576590, 0.621290, 1.043860, -0.328150, 1.029120, 0.219760, 0.451790, 0.498250, -0.128120, 0.309100, 
0.318860, 0.562230, 0.393610, -0.032720, 0.088560, -0.561200, 0.304540, -0.249920, -0.302740, -0.000540, 
0.022120, -0.176930, -0.356500, -0.050140, 0.005040, 0.506000, -0.683570, 
-0.208760, -0.242390, 0.552970, -0.152160, 0.223770, 0.117460, 0.105760, 0.137010, -0.003330, 0.362090, 
-0.218460, 0.376840, 1.192790, -0.346250, -0.110110, 0.870940, -0.190110, -0.449220, 1.070610, 0.349770, 
-0.820040, 0.691280, 0.622710, -0.928790, -0.153630, 0.636490, -0.069140, -1.238260, 0.355240, 1.458490, 
-0.639280, 0.734640, 0.725820, -0.242200, 0.726120, 0.332900, -0.132950, 1.014250, 0.254980, -0.217410, 
1.184830, 0.273850, -0.381440, 0.865240, 0.542360, -0.651070, 0.764230, 0.707380, -0.300430, -0.224320, 
0.561260, 0.327280, -0.631190, 0.105950, 0.969170, -0.385730, -0.566440, 
-0.737220, 0.381090, -0.801800, -1.129980, 0.235640, -0.254330, -1.198980, 0.720350, -0.799960, -1.173010, 
0.650280, -1.195990, -1.676400, 0.994790, -0.605880, -1.601740, 1.187690, -1.067220, -1.532230, 1.781600, 
-1.587890, -0.820700, 0.337610, -0.870450, -0.276620, -1.060920, 0.324230, -0.554510, -1.314180, 1.023830, 
0.092060, -0.293200, -0.685650, 0.679580, -0.184280, -1.128700, 0.049030, -0.428970, -1.233100, 0.030030, 
-0.011530, -1.078290, -0.515800, -0.014760, -0.643960, 0.109950, -0.781720, -0.041620, 0.318870, -0.679450, 
-0.285640, -0.433050, 0.034150, -0.117370, -0.879370, 0.534490, -0.223410, 
0.792560, -0.118860, 0.296270, 0.541770, 0.266600, 0.153030, -0.312450, 0.221190, 0.302240, -0.372300, 
0.278870, -0.422430, -0.131740, -0.362730, -0.003210, -0.072980, -0.506390, -0.041260, -0.712970, 0.374260, 
-0.477960, -0.208940, 0.561780, -0.472170, -1.194010, -0.067740, 1.537200, -2.405290, -0.380760, 1.575880, 
-0.109280, -0.482660, 0.192410, 1.398560, -0.698480, -0.449500, 0.291240, -0.769860, -0.301380, -0.019760, 
0.452090, -1.140810, -0.165870, -0.372250, -0.212630, -0.185060, -0.819650, 0.265570, 0.360810, -0.621660, 
-0.275640, 0.080180, -0.287420, -1.078090, 0.400530, -0.426470, 0.483400, 
-0.554700, 0.221080, -0.082930, -0.463800, 0.397690, -0.233550, -0.073930, 0.100380, -0.419550, -0.300660, 
-0.051650, -0.479060, -1.029550, 0.152990, -0.036020, -1.304110, 0.297910, -0.515340, -0.755130, -0.214490, 
-0.360550, -0.689670, -0.381970, -0.362000, -0.072550, -0.305710, -0.711240, -0.961590, -1.803270, 1.639590, 
-0.849070, -0.143910, 0.795000, -0.789730, 0.566960, 0.322750, -0.698860, 0.168330, 0.430530, -0.594080, 
1.004240, 0.361520, -0.515220, 0.348490, 0.289490, -0.095910, 0.251290, -0.370220, -0.079900, 0.057750, 
0.050760, -0.419520, -0.270100, 0.298700, 0.093280, 0.422190, -0.201000, 
-0.405120, 1.465700, 1.957290, 1.132870, 1.910050, -2.626570, -0.648130, -1.349490, 0.327800, -2.432520, 
1.989890, 0.399690, -0.944930, -2.171210, -2.150010, -1.176640, -1.102570, -0.686820, -2.095280, -1.296950, 

2.828490, -1.562820, -1.269020, -1.052430, -2.114780, -0.542590, -1.459140, -1.196780, -2.146530, -0.721500, 
-0.145520, -2.281950, -2.232920, 1.274710, 3.100620, -2.071040, 2.516960, -2.789040, -2.404570, -1.552840, 

-0.871190, 1.764440, 0.863230, 0.177780, 0.085510, 1.251900, 1.725320, 1.423020, 1.532140, 2.812660, 
-0.870170, 0.304370, 0.735360, -1.232510, 1.308120, 2.152060, 0.073860, 2.493550, 2.478060, 2.363550, 


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
      0.0, -1.031270, 57,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 59 (Old: 59) */
      0.0, 1.229310, 57,
       &Sources[57] , 
       &Weights[57] , 
      },
    { /* unit 60 (Old: 60) */
      0.0, 1.000650, 57,
       &Sources[114] , 
       &Weights[114] , 
      },
    { /* unit 61 (Old: 61) */
      0.0, 0.965200, 57,
       &Sources[171] , 
       &Weights[171] , 
      },
    { /* unit 62 (Old: 62) */
      0.0, 0.313340, 57,
       &Sources[228] , 
       &Weights[228] , 
      },
    { /* unit 63 (Old: 63) */
      0.0, -1.343720, 57,
       &Sources[285] , 
       &Weights[285] , 
      },
    { /* unit 64 (Old: 64) */
      0.0, -0.002950, 57,
       &Sources[342] , 
       &Weights[342] , 
      },
    { /* unit 65 (Old: 65) */
      0.0, -0.576640, 57,
       &Sources[399] , 
       &Weights[399] , 
      },
    { /* unit 66 (Old: 66) */
      0.0, 1.082560, 57,
       &Sources[456] , 
       &Weights[456] , 
      },
    { /* unit 67 (Old: 67) */
      0.0, -0.605770, 57,
       &Sources[513] , 
       &Weights[513] , 
      },
    { /* unit 68 (Old: 68) */
      0.0, 1.162130, 57,
       &Sources[570] , 
       &Weights[570] , 
      },
    { /* unit 69 (Old: 69) */
      0.0, 0.471190, 57,
       &Sources[627] , 
       &Weights[627] , 
      },
    { /* unit 70 (Old: 70) */
      0.0, -0.276060, 57,
       &Sources[684] , 
       &Weights[684] , 
      },
    { /* unit 71 (Old: 71) */
      0.0, -3.030030, 57,
       &Sources[741] , 
       &Weights[741] , 
      },
    { /* unit 72 (Old: 72) */
      0.0, -2.080910, 57,
       &Sources[798] , 
       &Weights[798] , 
      },
    { /* unit 73 (Old: 73) */
      0.0, -0.231150, 57,
       &Sources[855] , 
       &Weights[855] , 
      },
    { /* unit 74 (Old: 74) */
      0.0, -1.038240, 57,
       &Sources[912] , 
       &Weights[912] , 
      },
    { /* unit 75 (Old: 75) */
      0.0, -1.292810, 57,
       &Sources[969] , 
       &Weights[969] , 
      },
    { /* unit 76 (Old: 76) */
      0.0, -0.483870, 57,
       &Sources[1026] , 
       &Weights[1026] , 
      },
    { /* unit 77 (Old: 77) */
      0.0, -0.330740, 57,
       &Sources[1083] , 
       &Weights[1083] , 
      },
    { /* unit 78 (Old: 78) */
      0.0, -0.943730, 20,
       &Sources[1140] , 
       &Weights[1140] , 
      },
    { /* unit 79 (Old: 79) */
      0.0, 1.251140, 20,
       &Sources[1160] , 
       &Weights[1160] , 
      },
    { /* unit 80 (Old: 80) */
      0.0, -8.826760, 20,
       &Sources[1180] , 
       &Weights[1180] , 
      }

  };



int hmm2(float *in, float *out, int init)
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
