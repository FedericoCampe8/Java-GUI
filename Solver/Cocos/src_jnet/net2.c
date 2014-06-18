/*********************************************************
  net2.c
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:30 1999
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
  static UnitType Units[95];
  /* Sources definition section */
  static pUnit Sources[] =  {
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, 
Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, 
Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, 
Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, 
Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, 
Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, 
Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, 
Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76, 
Units + 77, Units + 78, Units + 79, Units + 80, Units + 81, Units + 82, Units + 83, Units + 84, Units + 85, Units + 86, 
Units + 87, Units + 88, Units + 89, Units + 90, Units + 91, 
Units + 77, Units + 78, Units + 79, Units + 80, Units + 81, Units + 82, Units + 83, Units + 84, Units + 85, Units + 86, 
Units + 87, Units + 88, Units + 89, Units + 90, Units + 91, 
Units + 77, Units + 78, Units + 79, Units + 80, Units + 81, Units + 82, Units + 83, Units + 84, Units + 85, Units + 86, 
Units + 87, Units + 88, Units + 89, Units + 90, Units + 91, 

  };

  /* Weigths definition section */
  static float Weights[] =  {
0.776990, -1.219010, -0.530380, -0.765810, 0.046350, -0.778850, 0.101840, -0.594630, -0.225160, -0.287880, 
0.676380, -1.078000, -0.857020, 0.596430, 1.105220, 0.042000, -0.816550, 0.462960, 0.581620, 0.283580, 
-0.135530, 0.506120, -0.178050, 0.937930, -0.275400, 0.945000, -0.660970, 1.350460, -1.653710, 2.110980, 
-0.903150, 1.085930, -2.386820, 2.106660, -0.728610, 0.866920, -2.426940, 0.673700, 0.150060, 0.616880, 
-1.795950, -0.138300, -0.053950, 0.018380, -1.165280, -0.321150, -0.399030, -0.918150, -0.913450, -0.879800, 
-0.222810, -1.217510, -0.825780, -0.428590, -0.060690, -1.055370, 0.049290, -0.082830, -0.385900, -0.659050, 
0.338480, -0.025360, -0.433220, -0.678230, 0.132960, 0.099200, 0.125970, 0.109510, -0.382440, 0.634650, 
0.166280, 0.311040, -0.246180, 0.551090, -0.067410, 0.148980, 
0.268390, -0.004100, 0.890000, 0.573910, 0.101070, 0.252830, 0.932370, 0.506300, 0.533230, 0.697100, 
-0.098850, 0.138840, 0.990060, -0.085580, -0.944550, -0.099520, 1.008310, -1.071160, -0.536490, -1.289310, 
0.955810, -0.961720, -0.283270, -0.491900, -0.300050, 0.002830, -0.431480, -0.788040, -0.046980, 0.758380, 
-1.557940, -0.950670, -0.328080, 2.262570, -2.225040, -0.491290, -1.857610, 1.399720, 1.389860, 1.252870, 
0.203260, 0.192240, 1.425550, 2.142850, -0.646610, 1.001610, 1.608890, -0.169340, -0.978240, 0.805330, 
1.875430, -1.370880, -0.183530, 1.016290, 1.277740, 1.828100, 0.743010, 0.940890, 1.042460, 0.879670, 
0.680820, 0.963040, 0.206980, -1.154820, -0.249260, 0.902480, 0.184140, -0.102230, 0.149120, -0.214210, 
-0.176510, 0.907780, -0.496900, 0.010560, -0.345520, 0.262800, 
0.074100, 0.985110, -0.131750, 0.390920, -1.245850, -0.139510, 0.552270, -0.137390, -0.536780, -0.215900, 
0.003500, 0.147110, -0.579460, 0.116080, -0.936450, 0.145930, -0.557470, 0.509120, -0.379020, 0.326340, 
-0.652700, 0.770670, -0.777480, -0.359170, 0.115480, -0.134560, -0.834240, -0.506220, 0.386140, -1.089540, 
1.350650, -0.677190, -0.499570, -0.751830, 2.126560, -0.446840, -1.140840, -1.274700, 3.300950, -0.276860, 
-0.304010, -0.066860, 0.774610, 0.040140, 0.222900, -0.289590, -0.063380, 0.582640, 0.332450, -0.408030, 
-0.560170, 0.235250, 1.517780, -0.899740, -0.374330, -0.583900, 0.603480, -0.482460, 0.293430, 0.445990, 
-0.505030, 0.011650, 0.209910, 0.409140, -1.056540, 0.450700, 0.315020, -0.579880, 0.002740, -0.191000, 
0.675560, -1.259040, 0.312890, -0.268870, 0.034320, -0.623430, 
-0.487700, 0.531460, -0.619060, -0.467260, -0.378980, -0.097980, -0.232080, -0.431630, 0.283940, -0.179810, 
-0.237820, -0.387110, -0.095160, 0.274690, 0.084760, 0.120910, -0.821670, 0.604290, 0.461740, 0.024970, 
-1.067960, 1.146390, 0.856690, 0.647750, -0.926430, 1.530930, 0.842620, 0.611580, -0.554960, 1.213440, 
1.225130, 1.393270, -0.254270, 1.538500, 1.152470, 1.814910, 0.440840, 1.150810, 1.258670, 0.847520, 
0.080370, 1.718080, 0.158440, -0.562440, -0.348200, 1.706940, -0.547010, -0.135790, -0.774380, 0.752550, 
-0.086690, -0.419300, -0.239460, -0.140780, 0.225070, -0.814100, 0.421690, -0.338820, 0.081370, -1.045370, 
-0.104180, 0.243370, -0.476010, 0.331030, 0.105910, 0.096760, -0.835540, -0.176040, -0.778050, 0.241110, 
-0.004830, -0.759120, -0.134570, -0.329990, 0.553370, -0.129730, 
0.787600, 0.439420, 0.288020, 0.371330, 0.606230, 0.201480, 0.703680, 0.248910, 0.669780, 0.047090, 
0.808920, 0.199840, 0.711690, 0.075700, 0.683860, -0.090610, 0.641680, 0.265560, 0.469050, -0.186950, 
0.484230, 0.607860, 0.167600, -0.091160, 0.187950, 0.883670, 0.092200, -0.074880, 0.066510, 0.925660, 
0.171310, -0.012280, -0.628880, 0.770040, 1.158670, 0.407310, -1.755460, 0.421970, 2.750630, 0.768010, 
-0.510720, 0.545510, 1.465740, 0.768280, 0.145890, 0.921070, 0.454330, 0.273950, 0.299010, 1.079140, 
0.069110, 0.013720, 0.494050, 0.915160, 0.074860, 0.633300, 0.682160, 0.563140, 0.282290, 0.399670, 
0.647590, 0.241400, 0.557030, -0.138780, 0.610090, 0.014760, 0.714700, 0.075470, 0.577660, -0.082240, 
0.787120, 0.272960, 0.533580, -0.027080, 0.715550, 0.016050, 
-0.135830, -0.205240, -1.207400, -0.611980, -0.478250, 0.150550, -0.338650, 0.202360, 0.202440, -0.022390, 
-0.192950, 0.154100, 0.376920, 0.650500, -0.755900, 0.790390, -0.006740, 1.018330, -0.458100, 0.874350, 
0.228460, 1.418730, -1.084440, 0.430950, 0.048120, 1.264910, -1.622070, 0.441830, -0.046240, 0.326710, 
-2.485150, 0.560540, -0.439790, -1.231930, -1.687400, 0.286190, -0.787280, -2.419780, 1.031320, -0.228730, 
0.151830, -2.098330, -1.685410, -1.575340, 0.370400, -1.621680, -1.652960, -1.700410, 0.194370, -0.967600, 
-0.777960, -1.995890, 0.409520, 0.028650, -0.492430, 0.088500, 1.118560, 0.165340, -1.037440, 0.111910, 
0.920100, -0.797140, -0.602120, -0.687310, 0.777740, -1.183300, -0.411800, 0.731650, 0.030560, -1.135330, 
-0.155830, 1.229420, 0.490790, -1.070120, -0.770260, -0.249870, 
0.597520, -0.389840, -0.757790, 0.258180, 0.738060, -0.448580, 0.414740, 0.750690, 1.683120, -0.230770, 
-0.077790, -0.243090, 0.735750, 0.501300, 0.092840, -1.246960, 0.535230, 0.800710, 0.367470, 0.070680, 
1.540890, -0.088900, 0.121270, 0.427220, 1.248860, -0.260450, -1.079360, -1.895310, -0.449030, -1.783420, 
-1.087180, -1.848940, -1.912620, -3.070450, -0.879770, -1.694070, -1.272320, -4.100030, 0.955530, -2.124020, 
-1.560540, -1.399750, -3.014290, -0.704380, -0.543530, 0.399180, -3.199580, 0.555040, -0.063300, 1.317620, 
-2.502850, 1.265690, -0.348320, 1.476970, -0.422900, 1.779390, 0.129280, 0.939120, -0.557670, 1.611370, 
-0.513120, -0.128780, -0.335180, 0.911870, -0.927650, -0.033320, -0.361560, 0.431470, -1.393680, 0.683270, 
-0.867810, -0.509390, -0.262240, -0.323100, 0.091210, -0.520760, 
-0.112730, 0.021040, 0.080080, -0.786180, -0.285300, -0.378560, 0.328430, -1.137560, 0.222430, -0.440380, 
0.229110, -0.454040, 0.267980, -0.266470, 0.093500, -0.756990, 0.154650, -0.033860, 0.272630, 0.172010, 
-0.099260, 0.064520, 0.915760, 0.448080, -0.162650, -0.099440, 1.367820, 1.260410, 0.055240, -0.343500, 
1.764710, 1.034360, -0.067490, -0.466150, 1.878740, 0.814410, -0.557280, -0.443610, 2.074070, 0.246920, 
-0.596520, 0.151680, 1.252360, -0.557270, -0.314240, 0.834930, -0.058360, 0.194390, -0.202080, 0.979520, 
-0.912450, 0.441340, 0.690140, 0.309020, -0.429720, -0.082300, 1.019300, -0.165640, -0.424930, 0.093790, 
0.684780, -0.234640, -0.378000, 0.482340, 0.334860, -0.482850, 0.107900, 0.108700, -0.009610, 0.019960, 
0.152070, -0.477430, -0.131280, 0.206150, 0.062370, -0.075440, 
-0.567900, 0.367790, 0.123310, 0.017140, -0.459400, 0.213350, 0.197060, -0.210010, -0.557510, -0.036130, 
0.404840, -0.035330, -0.709930, -0.401080, 0.336830, -0.466010, -1.060230, -0.126070, 1.135190, -0.137980, 
-0.952190, -1.098360, 1.462670, 0.033220, -1.496760, -1.648600, 1.090100, -0.155830, -1.733590, -2.399660, 
-0.459530, -0.016110, -1.683030, -2.313910, -2.370870, 0.399450, -1.777160, -3.304060, 0.878130, 0.312410, 
-2.664690, -2.270870, -1.122280, 0.040780, -2.490800, -1.926340, -0.229100, 0.403050, -1.980980, -1.261760, 
0.911790, 0.985280, -1.204340, -0.228860, 1.092800, 0.068770, -0.205650, 0.246770, 0.253850, -0.120110, 
-0.094170, -0.037330, -0.274550, 0.024720, -0.359190, -0.393300, -0.402230, -0.053780, -0.035700, -1.207660, 
-0.508740, -0.134730, 0.275340, -1.465580, -0.687560, 0.358520, 
-0.355060, -1.009510, 3.148480, -0.169250, -0.445480, -0.246900, 1.913540, -0.287580, -0.161520, 0.587690, 
0.635540, 0.273350, -0.248190, 1.807490, -0.885880, 1.401730, 0.116550, 2.792850, -1.462710, 1.811270, 
-0.096200, 2.492620, -1.308730, 1.604470, -0.479730, 1.433220, -0.550850, 1.161210, -0.718210, -0.374570, 
0.497110, 0.579620, -0.866590, -1.786190, 1.435970, -0.415950, -0.559810, -2.441320, 1.873060, -0.700130, 
0.928820, -1.456220, -0.249310, -1.336620, 0.421300, -0.377840, -0.118360, -2.056950, -0.011950, 0.761260, 
0.085650, -1.318620, 0.297400, 1.453580, 0.032630, -0.691360, 0.456850, 1.935670, -0.352330, -0.415630, 
0.994190, 1.058940, 0.163390, 0.057770, 0.515150, 1.149120, 0.747210, 0.478590, 0.639420, 0.940740, 
0.959300, 0.315870, 0.178160, 0.603980, 1.145990, -0.401160, 
0.038960, 0.129250, 0.145570, 0.285250, 0.893020, 0.526330, 0.108520, -0.088360, 0.678250, 0.537770, 
1.011280, 0.283460, 0.021470, 0.118450, 0.505900, 0.280020, 0.418300, 0.289480, 0.195580, 0.457910, 
0.575740, 0.282770, -0.178290, 0.257500, 0.645420, 0.167330, 0.208950, 0.159840, -0.174930, -0.760450, 
-0.090040, -0.414250, -2.116960, -2.204850, -0.351600, -0.377510, -1.650600, -1.924120, 4.975980, -0.896230, 
-2.992120, -3.473000, -1.376540, -0.917890, -1.790090, -2.717090, -1.612930, -0.386600, -0.378620, -1.068820, 
-0.889180, 0.081660, 0.394250, 0.544950, 0.250690, 0.024100, 0.884890, 0.448840, 0.317360, -0.196180, 
0.260990, 0.009490, 0.185110, -0.162670, 0.067430, -0.295410, 0.214530, 0.459920, -0.539890, -0.677280, 
-0.508950, 0.940850, -0.166820, -0.557210, -0.117370, 0.455940, 
-0.681080, 0.291670, 1.206410, -0.191660, -0.435850, 0.245110, 0.425860, -0.128170, -0.290580, 0.463860, 
-0.081730, -0.287250, 0.210460, 0.400560, -0.584000, 0.413280, -0.344910, -0.050750, -0.137600, 0.363330, 
-1.120100, 0.112930, 0.580010, 0.291560, -2.236320, 1.440340, 0.500570, -0.864990, -2.741380, 2.157140, 
0.490050, -0.623680, -1.431900, 1.539100, 0.051610, -0.276640, 0.274730, 0.100790, 0.461790, 0.866450, 
-1.227050, 0.703070, 0.482550, 1.517050, -2.162190, 0.800160, 0.897980, 0.618230, -2.494390, -0.152490, 
1.417160, 0.565990, -2.202210, -0.085860, 1.227190, -0.436410, -1.375980, 0.589140, 0.506300, -0.182000, 
0.011900, 0.241920, -0.882320, -0.026170, -0.048000, 0.696610, -1.436970, -0.401020, -0.566820, 0.576820, 
-0.854560, -0.789590, -0.993240, 0.851920, -0.897000, -0.215590, 
0.589450, 1.016490, 0.706400, -0.037660, 0.880600, -0.476740, -0.709930, -0.003040, 0.635210, -0.855690, 
-0.221680, -0.494610, 0.301500, -1.111670, -0.423850, 0.833000, -0.508620, 0.304730, 0.922930, -0.130460, 
0.338530, -0.153600, 1.628220, 0.439940, 0.210630, -0.005010, -0.286250, 1.019310, -0.123290, -0.507060, 
-1.745040, 0.630350, -0.019980, -2.670840, -1.220240, 0.255660, 0.169020, -3.391310, 3.345600, -0.441430, 
-1.456380, -2.054080, 0.021290, -0.823900, -0.327630, -1.168850, -1.638530, -0.538200, -0.065870, 0.104230, 
-0.408530, -0.588390, 0.127550, 1.153530, 0.638210, 0.283270, -0.271810, 0.983490, 1.002430, 0.367780, 
0.615890, -0.005120, 0.168010, -0.163030, 0.527590, -0.162820, -0.337140, 0.113610, 0.452750, -0.304280, 
0.330230, 0.394590, 0.713280, -0.614400, -0.405180, 0.759860, 
1.346600, 0.132090, 0.875110, -1.123250, 0.998330, -0.347520, 0.732060, -1.264090, 1.234580, -0.677340, 
0.242840, -0.435190, -0.598320, -0.378690, 0.312460, -0.323660, -0.745970, 0.572570, 0.483080, 1.356720, 
-0.663810, 1.377600, -0.031300, 1.422400, -0.079080, 1.517590, -1.138510, 0.678880, -0.009610, 0.393980, 
-1.768520, 0.950990, 0.074670, -0.663610, -1.973050, 0.937850, 0.109090, -2.768020, 2.107070, 0.856600, 
-0.268650, -2.247510, 1.405810, 0.266540, -0.636490, -1.648470, 1.145480, 0.144540, -0.380380, -1.148910, 
1.348080, -0.782360, 1.086540, -1.793710, 1.599600, -1.053310, 1.285190, -1.581070, 0.849960, -1.522720, 
0.981130, -0.076870, -0.493740, -1.238980, 0.385930, 1.359460, -0.665230, 0.241270, -0.070560, 2.074300, 
-0.392130, 0.695360, 0.233300, 1.498810, 0.015700, 0.574520, 
1.119280, 0.343850, -0.262970, -0.086280, 0.240740, -0.163780, 0.464460, 0.186220, 0.351180, -0.136480, 
0.426560, 0.382560, -0.489060, 0.298370, 0.502810, 0.101270, 0.012280, 0.163660, 0.344590, 0.089150, 
-0.337020, 0.554630, -0.256460, -0.426980, -0.509550, -0.194640, 0.095220, -0.491170, -0.519150, -0.558100, 
0.408260, -0.496050, -1.073060, -0.569820, 1.002170, -0.404830, -1.069590, -1.855790, 2.099290, -0.251400, 
-0.780780, -1.644990, 2.008440, 0.318550, -0.513110, -1.224170, 1.588140, -0.518420, -0.615310, -0.306090, 
1.117700, -0.224560, -0.527960, 0.595570, 1.054230, 0.078540, 0.105930, 0.667620, 0.644140, 0.013760, 
0.043480, 0.681540, 0.316920, 0.366770, -0.041730, 0.716150, -0.069760, 0.473540, 0.387070, 0.235340, 
-0.034890, 0.601130, 0.306660, -0.281140, -0.004370, 0.399430, 
-2.744190, -2.230540, -0.740020, -1.179310, -1.241110, -1.843930, -3.792320, -1.534190, -2.188600, 3.352190, 
-2.186040, -1.379910, 0.750100, 2.344120, -1.853780, 
1.121070, 2.791580, -0.523970, 3.607830, 0.241430, -2.048150, -0.858180, -0.749370, -2.392780, -2.047320, 
-1.916960, 0.109340, -1.405130, -1.174650, -0.516370, 
-0.271410, 1.345430, 0.960230, 0.367590, 0.628900, 1.978850, 3.515230, 2.440210, 3.406080, 0.626160, 
2.017690, 0.708380, 1.337740, 0.146660, 2.905830, 

  };

  /* unit definition section (see also UnitType) */
  static UnitType Units[95] = 
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
      0.0, 0.004760, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 59 (Old: 59) */
      0.0, -0.000920, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 60 (Old: 60) */
      0.0, 0.002950, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 61 (Old: 61) */
      0.0, -0.003780, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 62 (Old: 62) */
      0.0, 0.004410, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 63 (Old: 63) */
      0.0, 0.002390, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 64 (Old: 64) */
      0.0, 0.001970, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 65 (Old: 65) */
      0.0, -0.003910, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 66 (Old: 66) */
      0.0, 0.000020, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 67 (Old: 67) */
      0.0, -0.002100, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 68 (Old: 68) */
      0.0, 0.002140, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 69 (Old: 69) */
      0.0, -0.003650, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 70 (Old: 70) */
      0.0, -0.001760, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 71 (Old: 71) */
      0.0, 0.000420, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 72 (Old: 72) */
      0.0, -0.004600, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 73 (Old: 73) */
      0.0, -0.002750, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 74 (Old: 74) */
      0.0, 0.001230, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 75 (Old: 75) */
      0.0, 0.002710, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 76 (Old: 76) */
      0.0, 0.001450, 0,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 77 (Old: 77) */
      0.0, -1.156410, 76,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 78 (Old: 78) */
      0.0, 0.684730, 76,
       &Sources[76] , 
       &Weights[76] , 
      },
    { /* unit 79 (Old: 79) */
      0.0, 1.681820, 76,
       &Sources[152] , 
       &Weights[152] , 
      },
    { /* unit 80 (Old: 80) */
      0.0, 0.614740, 76,
       &Sources[228] , 
       &Weights[228] , 
      },
    { /* unit 81 (Old: 81) */
      0.0, 1.400380, 76,
       &Sources[304] , 
       &Weights[304] , 
      },
    { /* unit 82 (Old: 82) */
      0.0, -0.805890, 76,
       &Sources[380] , 
       &Weights[380] , 
      },
    { /* unit 83 (Old: 83) */
      0.0, -0.252200, 76,
       &Sources[456] , 
       &Weights[456] , 
      },
    { /* unit 84 (Old: 84) */
      0.0, 0.514860, 76,
       &Sources[532] , 
       &Weights[532] , 
      },
    { /* unit 85 (Old: 85) */
      0.0, -1.026160, 76,
       &Sources[608] , 
       &Weights[608] , 
      },
    { /* unit 86 (Old: 86) */
      0.0, 1.617830, 76,
       &Sources[684] , 
       &Weights[684] , 
      },
    { /* unit 87 (Old: 87) */
      0.0, -0.409730, 76,
       &Sources[760] , 
       &Weights[760] , 
      },
    { /* unit 88 (Old: 88) */
      0.0, -1.326810, 76,
       &Sources[836] , 
       &Weights[836] , 
      },
    { /* unit 89 (Old: 89) */
      0.0, 3.027250, 76,
       &Sources[912] , 
       &Weights[912] , 
      },
    { /* unit 90 (Old: 90) */
      0.0, 2.838970, 76,
       &Sources[988] , 
       &Weights[988] , 
      },
    { /* unit 91 (Old: 91) */
      0.0, 1.396410, 76,
       &Sources[1064] , 
       &Weights[1064] , 
      },
    { /* unit 92 (Old: 92) */
      0.0, 1.861810, 15,
       &Sources[1140] , 
       &Weights[1140] , 
      },
    { /* unit 93 (Old: 93) */
      0.0, -1.577090, 15,
       &Sources[1155] , 
       &Weights[1155] , 
      },
    { /* unit 94 (Old: 94) */
      0.0, -10.649180, 15,
       &Sources[1170] , 
       &Weights[1170] , 
      }

  };



int net2(float *in, float *out, int init)
{
  int member, source;
  float sum;
  enum{OK, Error, Not_Valid};
  pUnit unit;


  /* layer definition section (names & member units) */

  static pUnit Input[76] = {Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57, Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, Units + 67, Units + 68, Units + 69, Units + 70, Units + 71, Units + 72, Units + 73, Units + 74, Units + 75, Units + 76}; /* members */

  static pUnit Hidden1[15] = {Units + 77, Units + 78, Units + 79, Units + 80, Units + 81, Units + 82, Units + 83, Units + 84, Units + 85, Units + 86, Units + 87, Units + 88, Units + 89, Units + 90, Units + 91}; /* members */

  static pUnit Output1[3] = {Units + 92, Units + 93, Units + 94}; /* members */

  static int Output[3] = {92, 93, 94};

  for(member = 0; member < 76; member++) {
    Input[member]->act = in[member];
  }

  for (member = 0; member < 15; member++) {
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
