/*********************************************************
  net2b.c
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:32 1999
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
-0.176010, 0.097910, -0.212590, -0.533730, -0.370900, 0.268040, 0.068790, -0.777610, -0.152250, 0.139940, 
0.077500, -0.456370, -0.207130, 0.068840, -0.183370, 0.136220, -0.238830, 0.236070, -0.110720, 0.499580, 
-0.489030, 0.359230, -0.067440, 0.270340, -0.655890, 0.447730, -0.002920, -0.158060, -1.041990, 0.470850, 
0.210430, -0.352150, -1.896250, 0.255540, 0.411160, -0.045970, -2.412720, 0.697300, 0.796580, 0.318440, 
-2.723070, 0.594920, 0.193280, 0.170060, -1.825410, 0.403580, -0.087990, 0.039660, -1.041370, -0.126410, 
-0.313970, 0.048880, -0.570390, -0.153560, -0.326590, 0.168710, -0.277110, 0.042690, -0.349170, 0.067580, 
-0.021710, -0.155590, -0.270470, 0.019850, 0.010490, -0.380880, -0.129020, 0.057380, -0.159200, -0.416770, 
-0.110130, 0.101250, -0.880970, 0.034600, -0.205070, 0.162290, 
-0.355900, -0.178900, -0.684980, -0.175530, -0.326130, -0.535220, -0.327810, 0.359350, -0.592980, 0.120080, 
-0.146540, 0.002440, -0.889160, -0.094730, 0.632650, 0.186380, -1.092530, 0.529670, 0.607050, 0.724550, 
-0.838720, 0.328630, 0.833880, 1.205120, -0.777540, 0.072780, 1.440460, 1.000710, -0.546600, -0.013080, 
1.603390, -0.129800, -0.434300, 0.438100, 1.430230, 0.429360, -0.833440, 0.735010, 1.101330, 0.859240, 
-0.459740, 0.984780, 0.250580, 0.197680, 0.030860, 1.654550, -1.061750, -0.580420, 0.283190, 1.175360, 
-0.731730, -0.083320, 0.839540, 0.018580, -0.038000, -0.025420, 1.019200, -0.392540, 0.282680, -0.841710, 
0.295330, -0.033140, 0.166110, -0.062450, 0.238990, 0.490460, 0.067680, -0.049980, -0.073840, 0.668890, 
0.574700, -1.213450, -0.208290, -0.247810, 0.804360, -1.074660, 
-0.056850, 1.358910, -0.580230, 0.495790, 0.098510, 0.140580, 0.167870, -0.017810, 0.579370, -0.367910, 
0.077290, 0.194780, 0.601420, -0.570000, 0.114710, 0.195690, 0.585110, -0.177140, 0.021910, -0.482900, 
0.522430, -0.078630, -0.086760, -0.207550, 0.644330, 0.102540, -0.510360, -0.073810, 0.345490, 0.262050, 
-0.247400, -0.499390, 0.091430, 0.061170, -0.068280, -0.855570, -0.034620, -0.071430, 0.474560, -0.505520, 
-0.260750, 0.283710, 0.906050, 0.021580, -0.588400, 0.492600, 1.056230, 0.182200, -1.732070, 0.475210, 
1.274450, -0.300170, -2.084210, 0.725450, 0.958400, -0.363590, -1.742540, 0.656570, 0.770180, -0.225000, 
-1.019630, 0.623800, 0.277980, -0.411940, -0.496070, -0.100260, 0.532740, -0.044550, -0.305700, 0.095070, 
0.208540, 0.305910, -0.276920, 0.184070, -0.128830, 0.257180, 
-0.851790, 0.456170, -0.105670, -0.604610, -0.544530, -0.220500, 0.041130, -0.842070, -0.154840, -0.483100, 
-0.233660, -0.185090, 0.083570, -0.503190, -0.192740, 0.145820, -0.245890, 0.184640, -0.252760, 0.290440, 
-0.691150, 0.457470, -0.047240, 0.023510, -0.626080, 0.170890, 0.302670, -0.000460, -0.695280, 0.038940, 
0.975270, 0.579840, -0.324950, -0.382480, 1.343560, 0.749670, -0.197400, -1.326290, 1.702770, -0.545640, 
-0.298790, -0.556250, 1.551480, -0.880260, -0.315920, 0.339890, 0.720610, 0.251490, -0.618960, 0.488320, 
0.335280, -0.038300, -0.283630, 0.302860, -0.115520, -0.441540, 0.007950, 0.315250, -0.164270, -0.010160, 
0.467330, -0.079340, 0.133480, 0.342630, 0.309110, -0.122280, 0.264700, 0.089930, -0.070230, 0.160820, 
0.289210, 0.391460, 0.080990, 0.069660, 0.404090, 0.473900, 
0.783550, -0.048460, -0.258880, -0.111670, 0.537250, 0.177570, -0.008160, 0.232190, 0.430840, 0.103610, 
0.415970, 0.098710, 0.218100, 0.647040, 0.184880, 0.046950, 0.378600, 0.236590, -0.105830, -0.486750, 
0.705110, -0.382500, 0.051390, 0.079800, 0.756060, -0.607170, -0.474360, 0.294280, 0.202670, -0.053180, 
-0.828210, -0.348010, -0.427470, 0.162870, -0.620420, -1.240390, -2.526600, 0.008760, 1.517750, 0.927060, 
0.305980, 0.078550, 0.826510, 2.198850, 0.282640, 0.279100, 1.645010, -0.076590, -0.401370, -0.199590, 
1.772700, -1.161380, -0.360230, -0.112390, 1.124040, 0.682230, -0.416870, 0.167460, 0.815820, 0.062950, 
-0.503800, 0.539220, 0.374670, -0.755550, -0.507300, 0.531420, 0.028950, -0.395560, -0.221990, 0.231250, 
-0.060310, 0.474280, -0.090680, -0.424250, 0.418010, 0.093830, 
0.049360, -0.391500, -0.350800, -0.481920, 0.271790, -0.257560, -0.328590, -0.171770, 0.062890, -0.287110, 
-0.002420, -0.242710, 0.381130, -0.747440, -0.365510, -0.047250, 0.503440, -0.552840, -0.504800, 0.555800, 
0.998340, -0.851140, -0.373080, 0.497090, 0.829040, -0.308650, -0.687970, -0.195250, 0.114120, -1.080540, 
-0.261510, -0.092890, -0.489930, -2.013510, 0.178480, -0.742760, -1.151250, -1.769310, 1.057730, -0.014760, 
-1.624550, -0.813080, -0.233220, -0.073730, -0.062960, 0.318540, -1.054910, 0.385720, -0.276240, 0.758010, 
-0.617720, 0.674020, -0.526260, 0.554580, 0.054360, 0.376380, -0.485540, 0.390060, 0.069600, 0.204720, 
-0.570100, 0.156580, -0.206110, 0.025970, -0.264310, 0.128470, -0.224560, -0.766090, -0.113820, 0.211180, 
-0.201600, -0.449520, -0.185900, -0.639710, -0.263470, -0.010240, 
-0.459520, -0.801240, -0.106060, 0.163630, 0.252160, -0.108360, 0.187510, -0.067720, 0.049570, 0.562240, 
0.275780, -0.014370, 0.019780, 0.239780, -0.496880, 0.447320, 0.423780, -0.020790, -0.254170, 0.200060, 
0.881140, 0.046350, 0.389460, -0.560830, 0.944440, -0.278060, 0.477950, -0.256160, -0.138670, 0.210840, 
-0.943470, -0.169590, -1.464650, -1.221010, -1.439450, 0.822040, -2.895950, -1.970490, 0.748640, 0.137810, 
-0.930840, -2.210390, -0.958490, -0.479760, 1.076470, -0.630160, -0.119870, -0.855060, 0.926170, -0.419650, 
0.437370, -1.219390, 1.177710, -0.892210, 0.602930, -0.191310, 0.941300, -0.666180, 0.078510, -0.184170, 
-0.707320, 0.113460, -0.406330, -0.370440, -0.212310, -0.314010, -0.528120, 0.414960, 0.115620, -0.386960, 
-0.525660, 0.422550, -0.551610, -0.773230, -0.712310, 0.663220, 
-0.174000, 0.569440, -0.145360, -0.261070, -0.264590, 0.063850, 0.194410, -0.457600, 0.099100, -0.370530, 
0.157710, -0.259060, 0.041580, -0.541730, 0.136250, 0.061460, -0.042290, -0.103750, -0.239550, 0.296420, 
-0.085690, 0.272650, -0.445270, 0.219870, -0.241620, 0.320760, -0.164520, 0.058400, -0.567700, 0.277070, 
0.813000, 0.395090, -0.573200, -0.070550, 1.564740, 0.683180, -0.442850, -0.602660, 1.856620, -0.227740, 
-0.284740, -0.323060, 1.710280, -0.277640, -0.577310, 0.284110, 1.270110, 0.052630, -0.819530, 0.584550, 
0.667750, -0.027530, -0.505820, 0.673950, 0.027040, -0.268680, -0.067690, 0.636820, -0.206230, -0.069690, 
0.395400, 0.463110, -0.192720, 0.069950, 0.694520, 0.213210, -0.237180, 0.135720, 0.611160, 0.236220, 
-0.291380, 0.296660, 0.576720, 0.071830, -0.267300, 0.493780, 
0.084150, 0.835760, -0.406270, 0.538710, -0.230690, 0.099910, -0.070300, -0.246360, 0.060360, -0.180630, 
0.137230, -0.198060, -0.104840, -0.483960, 0.405350, -0.474120, 0.064920, -0.259970, 0.643640, -0.451050, 
0.130330, -0.495260, -0.011560, -0.769220, -0.034710, -0.521080, -1.300770, 0.064070, -0.341270, -0.601270, 
-0.731330, 0.268330, -0.466630, -0.994690, 0.556490, 0.288810, -0.845040, -2.183380, 2.901190, 0.818500, 
-0.556620, -1.358350, 1.534880, 0.331330, -0.552270, -0.532170, 0.579680, 0.952580, -1.264120, -0.074920, 
0.337020, 0.481980, -0.953200, 0.637630, -0.349170, 0.200420, -0.347170, 0.417950, 0.114080, -0.086440, 
-0.071210, 0.233530, 0.264750, -0.227450, 0.368640, -0.722390, 0.829690, -0.190220, 0.253300, 0.020470, 
0.401390, 0.267490, 0.075740, -0.267900, 0.375000, 0.129990, 
-0.822490, -0.093060, 2.106550, 0.087410, -0.581260, 0.341070, 1.293820, 0.015560, -0.125670, 0.656680, 
0.308940, 0.333050, -0.079470, 1.510030, -0.635220, 0.877300, 0.204800, 1.215730, -0.876430, 0.581070, 
0.639300, 0.464560, -0.944540, 0.197300, 0.584480, -0.273810, 0.004980, 0.291600, 0.099370, -0.469710, 
0.796600, 0.162570, -0.055820, -1.159870, 1.602840, -0.074300, -0.295180, -1.274050, 2.019930, 0.107620, 
-0.279840, 0.097280, 0.030160, -0.675630, -0.101920, 0.735390, -0.060490, -0.457550, 0.048960, 0.587370, 
0.586440, -0.622770, 0.255090, 0.628140, 0.417460, -0.400790, 0.181300, 0.748910, 0.315290, -0.191170, 
0.575080, 0.527130, 0.285820, 0.368740, 0.578220, 0.420160, 0.373980, -0.224550, 0.103340, 0.739030, 
0.258280, 0.034710, -0.241940, 0.938610, 0.381010, -0.009080, 
0.054520, -0.255830, 0.534330, 0.906030, 0.108980, -0.007710, -0.047490, 0.534020, 0.383370, -0.537350, 
-0.214450, 0.088530, -0.217260, -0.361020, -0.340360, -0.354520, -0.077640, -0.648330, 0.349250, 0.226960, 
-0.229000, -0.195740, 0.331110, 0.423280, -0.573950, 0.279640, -0.219840, 0.108760, -0.949600, -0.328520, 
-0.159980, -0.148080, -1.906760, -1.709480, -0.879830, -0.459080, -1.732690, -2.459650, 1.627570, -0.281010, 
-2.557460, -1.237410, -1.084200, -0.566400, -1.479420, -0.639950, -0.398520, -0.717290, -0.425450, -0.002890, 
-0.053470, 0.296010, 0.051660, 0.196650, 0.007260, 0.278920, -0.129860, 0.179170, 0.403490, 0.432650, 
-0.313490, -0.176070, 0.083010, -0.197650, -0.045140, 0.320340, -0.339590, -0.216010, 0.171750, 0.079670, 
0.517540, -0.401960, 0.040940, -0.216060, 0.527680, 0.250200, 
-0.666640, -0.400120, -0.004060, 0.452390, -0.409940, -0.119830, -0.163460, 0.613750, 0.053510, -0.049100, 
0.120270, 0.290690, -0.114420, 0.186880, 0.189590, -0.247410, -0.202950, 0.505840, 0.371530, -0.700840, 
-0.409450, 0.838170, 0.340570, -0.398740, -0.781790, 0.629050, 0.349830, -0.424790, -0.859760, 0.224720, 
-0.010730, 0.040820, -1.335530, 0.879250, -0.374390, 0.324560, -1.530280, 1.234020, -0.481480, 1.150590, 
-1.412910, 1.410770, 0.036030, 0.839910, -1.110950, 1.213610, 0.522390, 0.375660, -0.871310, 0.355420, 
-0.254450, 0.437120, -0.513740, -0.282930, -0.533850, 0.206390, -0.235910, -0.294870, -0.384990, 0.062940, 
0.199010, -0.195400, -0.102110, 0.226250, 0.210100, 0.083800, 0.420780, 0.018450, 0.122540, -0.018080, 
0.210380, -0.271580, -0.712390, 0.082100, -0.218330, -0.780950, 
0.216500, 1.225960, -0.400560, -0.589850, -0.067450, 0.494350, 0.369800, -0.388370, -0.033780, 0.348060, 
0.589120, 0.071400, -0.439820, 0.109210, 0.020760, 0.095360, -0.558870, 0.518580, -0.384150, -0.464860, 
-0.876630, 0.463640, -0.304970, -0.462610, -0.760480, 0.154810, 0.216910, 0.175580, -0.297280, -0.040590, 
0.335710, 0.341980, -0.695590, -2.160040, 0.180940, 0.652120, 0.920760, -3.472550, 1.328840, -0.130880, 
-0.537350, -2.036150, -0.189180, 0.528700, -0.700620, -1.403590, 0.408530, 1.085710, 0.629050, -0.851590, 
-0.524720, 0.640360, 0.463360, 0.469200, -0.677900, -0.464450, 0.185450, 0.442130, 0.038890, -0.268830, 
0.345080, -0.193570, -0.331310, 0.206950, -0.617070, 0.207200, -0.259860, -0.125140, -0.604940, 0.123880, 
-0.870540, -0.039680, 0.569200, -0.662380, -1.062700, -0.667320, 
0.861670, -0.047940, 0.634140, -0.810780, 0.793670, -0.258190, 0.319380, -0.506020, 0.669250, 0.401670, 
0.891230, -0.481190, 0.673220, 0.815770, 0.547820, 0.405810, 0.691450, 1.248020, 0.323220, 0.127980, 
0.694370, 0.244170, 1.123840, 0.926360, 0.854560, -0.319800, 0.176390, -0.305390, 0.217830, -0.675440, 
-1.121900, -0.358500, -0.419510, -1.161260, -0.354610, 0.337210, -0.259430, -2.780860, 1.813690, -0.958580, 
-0.293620, -1.973010, -0.657580, 0.404630, -0.143010, -0.718880, -1.655460, 0.441200, -0.281520, -0.199840, 
-0.074230, 0.264280, 0.697640, -0.968660, 1.039040, 0.160150, 0.369140, -0.317280, 0.306350, 0.072650, 
0.987050, -0.670540, 0.030100, -0.200700, 0.873080, -0.980020, 0.326040, 0.518840, 0.706170, -0.180650, 
-0.071770, 0.414700, 0.715380, -0.127050, -0.069430, 0.317600, 
0.054880, 0.272120, 0.600130, -0.049220, 0.072640, 0.344650, -0.078240, -0.521140, 0.187570, 0.232860, 
-0.318850, -0.245020, -0.199660, 0.162900, -0.131760, -0.521450, 0.143680, -0.049920, 0.066870, 0.054580, 
0.076120, 0.038600, -0.039990, 0.315650, -0.461910, -0.147710, 0.563770, 0.077000, -0.720710, -0.502970, 
1.705400, 0.241630, -0.472090, -1.000480, 2.264350, 0.276380, -0.635370, -1.412650, 2.545410, -0.120950, 
-0.372290, -1.298570, 1.775660, -0.717810, -0.332360, -0.492080, 0.592610, -0.652640, -0.488520, -0.163980, 
-0.129060, 0.032030, -0.362490, 0.080600, -0.202030, -0.027920, 0.166690, 0.246510, -0.041370, 0.231900, 
0.407480, 0.442410, -0.098320, 0.549360, 0.619920, 0.101780, 0.180550, 0.369500, 0.474600, 0.173510, 
0.342870, 0.166550, 0.035760, 0.006780, 0.386590, 0.027970, 
-3.719030, -1.547060, -1.031600, -1.356980, -2.134010, -2.545520, -3.165850, -1.431990, 0.475320, 3.094730, 
-1.517560, -1.459790, -0.723360, 1.797770, -0.208620, 
0.005010, 1.697770, 0.568310, 2.649510, 1.895850, -1.463210, -0.767170, 1.182840, -1.615710, -2.580080, 
-2.338530, 1.474760, -1.374530, -1.727020, -1.288860, 
0.625530, 1.077030, 0.641030, -0.593610, 1.650350, 2.568220, 2.689570, 1.084340, 0.799810, 0.770080, 
2.452900, 0.436460, 1.497030, 1.427690, 2.197610, 

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
      0.0, 0.072880, 76,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 78 (Old: 78) */
      0.0, 2.332000, 76,
       &Sources[76] , 
       &Weights[76] , 
      },
    { /* unit 79 (Old: 79) */
      0.0, 1.658650, 76,
       &Sources[152] , 
       &Weights[152] , 
      },
    { /* unit 80 (Old: 80) */
      0.0, 1.923030, 76,
       &Sources[228] , 
       &Weights[228] , 
      },
    { /* unit 81 (Old: 81) */
      0.0, 1.965440, 76,
       &Sources[304] , 
       &Weights[304] , 
      },
    { /* unit 82 (Old: 82) */
      0.0, 0.100560, 76,
       &Sources[380] , 
       &Weights[380] , 
      },
    { /* unit 83 (Old: 83) */
      0.0, 0.821360, 76,
       &Sources[456] , 
       &Weights[456] , 
      },
    { /* unit 84 (Old: 84) */
      0.0, 1.637950, 76,
       &Sources[532] , 
       &Weights[532] , 
      },
    { /* unit 85 (Old: 85) */
      0.0, 1.017140, 76,
       &Sources[608] , 
       &Weights[608] , 
      },
    { /* unit 86 (Old: 86) */
      0.0, -0.311490, 76,
       &Sources[684] , 
       &Weights[684] , 
      },
    { /* unit 87 (Old: 87) */
      0.0, 0.907980, 76,
       &Sources[760] , 
       &Weights[760] , 
      },
    { /* unit 88 (Old: 88) */
      0.0, 1.607150, 76,
       &Sources[836] , 
       &Weights[836] , 
      },
    { /* unit 89 (Old: 89) */
      0.0, 1.039090, 76,
       &Sources[912] , 
       &Weights[912] , 
      },
    { /* unit 90 (Old: 90) */
      0.0, 1.397500, 76,
       &Sources[988] , 
       &Weights[988] , 
      },
    { /* unit 91 (Old: 91) */
      0.0, 1.186670, 76,
       &Sources[1064] , 
       &Weights[1064] , 
      },
    { /* unit 92 (Old: 92) */
      0.0, 2.739320, 15,
       &Sources[1140] , 
       &Weights[1140] , 
      },
    { /* unit 93 (Old: 93) */
      0.0, -2.557380, 15,
       &Sources[1155] , 
       &Weights[1155] , 
      },
    { /* unit 94 (Old: 94) */
      0.0, -9.705730, 15,
       &Sources[1170] , 
       &Weights[1170] , 
      }

  };



int net2b(float *in, float *out, int init)
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
