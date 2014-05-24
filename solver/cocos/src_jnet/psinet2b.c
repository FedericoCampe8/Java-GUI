/*********************************************************
  psinet2b.c
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:54 1999
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
  static UnitType Units[70];
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
Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, 
Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, 
Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66, 

  };

  /* Weigths definition section */
  static float Weights[] =  {
0.194090, -0.135900, -0.137230, -0.818210, -0.219440, -0.190870, -0.450850, -0.055620, -0.874850, -0.754430, 
-0.733710, 0.060430, -0.760700, -1.068970, -0.927980, -0.657850, -0.627040, -1.065690, -0.213290, -0.254170, 
-0.340930, -1.137680, -1.013280, -0.465790, 0.835100, -0.925790, -0.410900, -0.481480, -0.412670, 1.878030, 
0.978770, -0.172220, 2.708520, 0.212960, -0.329130, 1.634250, 1.268700, -0.161800, 1.092160, 0.477970, 
-0.261230, -0.088820, 0.660130, -0.369980, 0.061320, 1.089860, -0.067470, 0.156130, -0.070700, 0.384900, 
0.106810, 0.948840, 0.311080, 0.183220, 0.708760, -0.230810, -0.078130, 
0.575460, -0.597280, -0.442990, -0.455930, 0.095680, -0.317550, 0.184660, -0.490180, 0.295790, 0.619940, 
-0.276690, 0.092150, 0.988350, -0.023010, -0.009990, 1.374590, -0.722530, 0.135170, 1.222630, -1.989210, 
1.931520, -0.239910, -1.856540, 1.816210, -0.394910, -1.062520, 0.428630, 0.499120, -0.165040, -0.783240, 
1.482550, -0.537030, -1.395540, 1.456450, -1.414410, -1.091660, 1.234080, -0.045070, -0.577360, 0.377850, 
-0.419690, -0.352300, 0.288090, -0.162600, -0.076010, 0.268410, 0.576120, -0.033050, 0.276980, -0.118370, 
0.137360, -0.019200, 0.525920, -0.106740, 0.235510, -0.094810, 1.098130, 
-1.043960, 0.021060, -0.385480, -0.375080, -0.579740, -0.067040, 0.055470, -0.163590, -0.485080, 0.317840, 
0.405500, 0.269660, 0.739580, 0.787350, 0.219300, -0.429550, -0.092410, 0.418230, -0.482560, 0.571960, 
0.059450, -0.791850, 0.433740, -0.902310, -0.423550, -1.560790, -1.518080, -0.562220, -3.662430, -0.677230, 
-0.834120, 0.129740, -0.125760, -2.891390, 1.526090, 0.204030, -3.277600, 1.694870, 1.774440, -2.788490, 
1.264510, 0.533530, -2.789720, 0.732710, -0.213320, -0.318860, -0.307530, 0.152740, 0.470780, 0.361820, 
-0.797520, -0.692260, -0.483080, 0.147300, -0.988430, -0.430410, -0.988480, 
-0.079570, 0.219500, -0.637970, 0.862990, 0.049670, -0.618110, -0.196600, -1.528730, 0.790030, 0.589350, 
-1.960080, -0.279000, -0.359510, -1.745060, 0.222940, -0.066240, -0.458250, -0.587040, 0.310200, 0.785830, 
-1.524980, 0.288880, 0.497880, -1.877840, -1.129950, -0.591420, 0.605010, -3.336960, -1.645090, 2.662110, 
-0.175640, 0.854860, 0.675820, -0.241380, 0.782860, -0.023380, 0.044810, 0.028470, 0.567530, 0.351690, 
-0.874060, -0.762150, -0.018700, 0.098480, 0.453160, 0.255680, 0.540660, 0.542170, 0.445100, 0.432250, 
0.144750, 0.515400, 0.325680, 0.691210, -0.059330, 0.288580, -0.154380, 
-0.469010, -0.584980, -0.273770, -0.103920, 0.009730, -0.008000, 0.514480, 0.830130, 1.195100, 0.680420, 
0.445130, 1.145850, 1.132810, 0.614800, 1.124710, 0.449980, 1.019040, 0.038700, 0.203200, 1.262620, 
-0.148070, -0.155150, 0.991460, 0.335880, -0.214520, 0.989470, 1.273660, -1.742110, 0.243430, 4.199570, 
-0.857610, 0.053480, 0.530550, -1.222160, -0.595580, -0.693680, -0.274380, -0.694980, -0.258700, 0.343740, 
-0.207020, 0.212910, -0.001260, -0.028410, 0.113870, 0.307820, 0.643100, 0.610730, -0.165440, 0.731390, 
0.320580, -0.240420, -0.368990, -0.167560, 0.340040, 0.324060, 0.572600, 
0.083170, -0.288480, 0.333140, -0.067620, 0.272950, 0.235020, 0.103720, -0.253520, -0.438730, -0.400460, 
-0.235950, 0.188500, 0.553600, -0.043720, 0.040870, 0.638860, 0.352830, 1.004090, 0.489220, -0.709210, 
1.139440, -0.138720, -0.463990, 0.556730, -0.258520, 0.588990, 1.093300, -2.380660, 1.782900, 1.458870, 
-0.977400, 0.511850, 0.249270, -0.355010, 0.021950, 0.111690, -0.442280, 0.259480, -0.252190, 0.141900, 
0.142100, -0.178240, 0.621770, 0.181160, 0.515340, 0.591530, 0.658490, 0.649340, 0.312300, 0.359240, 
0.365910, 0.383690, 0.731060, 0.425170, 0.558840, 0.979540, 0.461950, 
0.111060, 0.091050, -0.069760, -0.100070, 0.203150, 0.466840, -0.068330, 0.283590, -0.009970, -0.215830, 
0.553530, 0.144740, 0.050300, 1.296970, 0.377630, -0.247250, 0.971760, 0.288720, -0.257930, 0.720610, 
0.180080, -0.506370, -0.672180, -0.305770, -0.328510, -0.456310, 0.138160, -1.206860, -1.855320, 1.399700, 
-0.568210, -1.801410, 0.382190, -0.140910, -1.308300, -0.963860, -0.257230, -0.849360, -0.527810, -1.188970, 
-0.215890, -0.324020, -0.464890, -0.098480, -0.170370, 0.278330, -0.072220, 0.147360, 0.047070, -0.411570, 
0.051300, 0.275470, -0.161210, -0.444700, -0.306740, 0.504420, 0.520400, 
-0.884010, 0.235270, -0.010890, -0.572930, -0.157650, 0.153790, -0.480950, 0.079610, -0.833200, -0.552810, 
1.237350, -0.322230, -0.301580, 0.206690, 0.944330, 0.437290, 0.559630, -0.719410, -0.759770, 1.362840, 
-1.004230, -2.032390, 0.382230, -0.828630, -0.856180, -2.693620, 0.491570, -2.896520, -1.624320, 2.564380, 
-1.347540, 0.620970, 0.224160, -0.833730, 0.552690, -0.996130, -0.470540, 0.804860, -1.726760, -0.119720, 
-1.033010, -0.353900, 1.084120, -1.083410, -0.046690, 0.341910, -0.120460, 0.531400, 0.141280, -1.022170, 
0.128440, -0.668430, -0.226560, -0.261880, -0.182400, -0.556280, -0.343840, 
0.292870, -0.941230, -0.181810, -1.021010, -0.904930, -1.040430, -1.075490, -0.750010, -0.942450, -0.371300, 
-0.738810, -0.766270, -0.097950, -1.210930, -0.448610, -0.291770, -1.839960, -0.738160, 0.230280, -2.323570, 
0.041860, 0.818690, -1.439470, 0.626650, 0.239080, -1.030170, -0.149210, 1.125760, -0.404880, -0.961210, 
1.423800, -0.807470, 0.663380, 1.231320, -1.625540, 0.970620, 0.516130, -0.610590, 0.286440, 0.210730, 
-1.025600, -0.467410, 0.289810, -1.286220, -1.051080, 0.146060, -0.220430, -0.632290, -0.378900, 0.252910, 
0.013990, 0.071050, 0.090080, 0.273790, 0.728640, -0.857040, -0.327180, 
2.397670, 3.691120, 0.676720, -1.185840, -3.684650, -2.630320, -0.937480, -3.114340, 0.582410, 
-2.807860, -2.130980, -3.277660, -1.585570, 2.342730, 0.749870, -2.188470, -2.023260, -2.778910, 
2.887170, 0.864710, 2.326060, 1.725260, 4.063460, 1.277420, 2.248350, 2.226710, 0.756580, 

  };

  /* unit definition section (see also UnitType) */
  static UnitType Units[70] = 
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
      0.0, 3.719740, 57,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 59 (Old: 59) */
      0.0, 4.212010, 57,
       &Sources[57] , 
       &Weights[57] , 
      },
    { /* unit 60 (Old: 60) */
      0.0, -0.384170, 57,
       &Sources[114] , 
       &Weights[114] , 
      },
    { /* unit 61 (Old: 61) */
      0.0, -1.601280, 57,
       &Sources[171] , 
       &Weights[171] , 
      },
    { /* unit 62 (Old: 62) */
      0.0, -1.512590, 57,
       &Sources[228] , 
       &Weights[228] , 
      },
    { /* unit 63 (Old: 63) */
      0.0, -5.656430, 57,
       &Sources[285] , 
       &Weights[285] , 
      },
    { /* unit 64 (Old: 64) */
      0.0, 0.069020, 57,
       &Sources[342] , 
       &Weights[342] , 
      },
    { /* unit 65 (Old: 65) */
      0.0, -0.645570, 57,
       &Sources[399] , 
       &Weights[399] , 
      },
    { /* unit 66 (Old: 66) */
      0.0, 5.132180, 57,
       &Sources[456] , 
       &Weights[456] , 
      },
    { /* unit 67 (Old: 67) */
      0.0, -2.182660, 9,
       &Sources[513] , 
       &Weights[513] , 
      },
    { /* unit 68 (Old: 68) */
      0.0, 2.661600, 9,
       &Sources[522] , 
       &Weights[522] , 
      },
    { /* unit 69 (Old: 69) */
      0.0, -9.335510, 9,
       &Sources[531] , 
       &Weights[531] , 
      }

  };



int psinet2b(float *in, float *out, int init)
{
  int member, source;
  float sum;
  enum{OK, Error, Not_Valid};
  pUnit unit;


  /* layer definition section (names & member units) */

  static pUnit Input[57] = {Units + 1, Units + 2, Units + 3, Units + 4, Units + 5, Units + 6, Units + 7, Units + 8, Units + 9, Units + 10, Units + 11, Units + 12, Units + 13, Units + 14, Units + 15, Units + 16, Units + 17, Units + 18, Units + 19, Units + 20, Units + 21, Units + 22, Units + 23, Units + 24, Units + 25, Units + 26, Units + 27, Units + 28, Units + 29, Units + 30, Units + 31, Units + 32, Units + 33, Units + 34, Units + 35, Units + 36, Units + 37, Units + 38, Units + 39, Units + 40, Units + 41, Units + 42, Units + 43, Units + 44, Units + 45, Units + 46, Units + 47, Units + 48, Units + 49, Units + 50, Units + 51, Units + 52, Units + 53, Units + 54, Units + 55, Units + 56, Units + 57}; /* members */

  static pUnit Hidden1[9] = {Units + 58, Units + 59, Units + 60, Units + 61, Units + 62, Units + 63, Units + 64, Units + 65, Units + 66}; /* members */

  static pUnit Output1[3] = {Units + 67, Units + 68, Units + 69}; /* members */

  static int Output[3] = {67, 68, 69};

  for(member = 0; member < 57; member++) {
    Input[member]->act = in[member];
  }

  for (member = 0; member < 9; member++) {
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
