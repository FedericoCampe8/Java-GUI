/*********************************************************
  psinet2.c
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:52 1999
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
-0.144520, 0.314260, 0.234400, -0.053270, 0.074850, 0.095650, 0.389370, 0.058310, 0.168800, 0.271910, 
0.493510, 0.135070, 0.412680, 0.160880, 0.574110, 0.808680, 0.700510, 1.136140, 1.095250, 0.119720, 
0.595320, -0.003820, -0.864280, -0.564420, -0.852810, -1.681980, -1.439330, -0.684890, -2.640330, -0.278200, 
-0.018990, -1.239780, -0.149360, -0.375770, -0.187180, 0.352160, -0.912530, 0.121070, 0.451960, -0.202290, 
0.767430, 1.104890, 0.365640, 0.451410, 0.472310, 0.115410, 0.590210, 0.070270, 0.022650, 0.063660, 
0.262050, 0.430620, -0.202230, 0.045370, 0.543230, 0.518990, 0.498700, 
0.169510, 0.934150, 0.259490, -0.021370, 0.400390, 0.992430, 0.309720, 0.364480, 0.504530, 0.839120, 
0.177240, 0.305700, 1.136700, 0.319550, 0.068920, 0.922110, 0.491740, 0.065720, 0.483250, -0.014410, 
0.709080, 0.204930, -0.112950, 0.596820, 0.114010, -1.029590, 0.795600, -0.810840, -2.210300, 2.610520, 
-0.584500, -1.661740, 1.763940, 0.210280, -1.662210, 1.494200, 0.708270, -0.839800, 0.872240, 0.908280, 
-0.322400, 0.296940, 1.101570, 0.194380, -0.231650, 1.053160, -0.430410, -0.009820, 0.667930, 0.439850, 
-0.293550, 0.426380, 0.802270, -0.653950, 0.417280, 0.308190, -0.333570, 
-0.333410, 0.349450, 0.464880, -0.156240, -0.129050, 0.699490, -0.009620, 0.404300, -0.281390, 0.114040, 
0.675130, -0.071760, -0.086460, 1.295780, 0.074680, -0.340320, 1.519000, -0.082470, -0.081590, 1.225690, 
-0.026450, 0.940780, 1.365600, -1.618070, 1.430610, 0.828850, -2.620780, 0.641670, -1.948140, 0.768870, 
0.164730, 0.442250, -1.107710, -0.214080, -0.475560, 0.646300, -0.168430, -1.053570, 1.953330, 0.150110, 
-0.454710, 1.582980, 0.479160, 0.355760, 0.523130, 0.412630, -0.022050, 0.426900, 0.217410, 0.134450, 
0.307240, 0.394870, 0.178210, -0.376200, 0.443690, 0.315010, -0.347480, 
1.325140, 0.859900, -1.195350, 0.889990, 0.492720, -0.176150, 0.901250, 0.124630, 0.136940, 0.966340, 
-0.055270, 0.374520, 1.080490, 0.136220, 0.223620, 1.039300, 0.154500, 0.031170, 1.037900, -0.733000, 
0.484950, 0.926910, -1.032190, 0.172700, 0.382480, -0.909700, -0.166420, -0.499960, -0.977740, 0.388900, 
-0.430890, -0.604140, 0.488480, 0.211030, 0.039790, 0.152820, 0.146410, 0.791880, -0.008520, 0.010090, 
0.502200, 0.255910, -0.230590, 0.530850, 0.368440, 0.137980, -0.363960, 0.502530, 0.210490, -0.242950, 
0.456240, -0.003430, -0.409010, 0.635950, -0.418920, -0.770480, 1.427480, 
0.028970, -0.294730, -0.124160, -0.330690, -0.316850, -0.361980, -0.175250, -0.055260, 0.050370, -0.212210, 
0.053550, -0.127250, -0.087270, -0.247310, -0.375550, -0.086010, 0.144450, 0.065970, 0.074110, 0.242730, 
0.251330, -0.236940, 0.219860, 0.261100, -0.054760, 0.481520, 1.381830, -0.690970, 1.553640, 2.887870, 
0.303870, 1.177670, 1.183780, 0.293790, 0.870130, 0.322220, -0.216990, -0.074830, -0.542880, -0.061480, 
-0.305790, -0.454170, 0.020590, -0.261040, -0.287100, 0.090680, 0.206140, 0.127620, -0.333690, -0.032370, 
0.040690, -0.045850, 0.120010, -0.110540, 0.486320, 0.935480, 0.850090, 
0.889960, 0.041910, 0.071840, 0.509700, 0.128720, 0.381070, -0.101040, 0.383230, 0.258010, 0.109300, 
0.240570, 0.182910, 0.199610, 0.061490, 0.350350, 1.020230, 0.501750, 0.627790, 0.055940, 0.608850, 
0.306160, -1.390060, 0.090580, 0.211570, -2.304990, -0.655310, -0.109200, -3.487520, -0.345250, 1.279680, 
-2.371710, 0.287890, 0.618740, -0.587060, -0.118260, -0.295360, -0.804380, -0.480320, -0.794880, -0.076350, 
-0.122760, -0.298390, -0.560410, -0.327580, -0.918790, -0.734890, -0.679540, -0.357530, 0.120470, -0.075490, 
0.019490, 0.152940, 0.251180, 0.190390, 0.035730, 0.188640, 0.116600, 
-0.318790, -0.015330, 0.050530, -0.483790, 0.506860, -0.670440, -0.664440, 0.590620, -0.252110, 0.098030, 
-0.667170, 0.285490, 0.086610, -0.344330, -0.073250, 0.163510, 1.665430, -0.434150, 0.777790, 1.377480, 
-0.363620, 0.437130, -0.156200, -0.765200, -0.491830, -1.267460, 0.516040, -2.854210, -2.230480, 3.431300, 
-0.103720, -0.295880, 0.012980, 1.053110, 0.321350, -2.049590, 0.807600, -0.288130, -1.309060, 0.768880, 
-0.950280, -0.352480, -0.227220, -1.201280, 0.036780, 0.281520, -1.249690, -0.276550, 0.056250, -0.249930, 
-0.555350, -0.025980, -0.230090, -0.091040, -0.684860, 0.204370, -0.126200, 
1.025210, 0.191770, 0.538370, 0.799380, 0.028900, 1.144350, 0.580050, 0.756410, 0.511320, 0.348840, 
0.578440, 0.827310, 0.774020, 0.100750, 0.712650, 1.038910, -0.804930, 1.365600, 0.750620, -0.925080, 
1.535820, 0.482330, -0.774810, 0.594530, 0.298070, -0.575910, -0.217260, -0.689510, -0.076710, 0.179700, 
-0.149570, 0.202840, -0.004970, 1.173310, 1.037820, -1.312780, 0.349320, 2.182100, -1.618110, -1.038810, 
2.815210, -1.130370, -1.504020, 2.923450, -0.966750, -0.387310, 1.322100, -0.620440, -0.276900, 1.416410, 
-0.519520, 0.261800, 0.594910, -0.071510, 0.439130, -0.352840, 0.994370, 
0.674280, -0.349350, -0.149720, 0.448260, -0.418670, -0.153460, -0.620280, -0.059290, 0.047700, -0.466790, 
0.465280, -0.238490, -0.118890, 0.878490, -0.225500, 0.704390, -0.168790, 0.097060, 0.276010, -0.676220, 
0.891690, -0.016790, -1.005820, 1.273100, -1.267790, 0.296560, 0.692130, -0.841000, 1.649120, -1.333580, 
0.027620, -0.288420, -0.488250, 1.083400, -1.928030, -0.274540, 1.433380, -1.649510, -0.928680, 0.856760, 
-0.214910, -1.268980, 0.335170, -0.917350, -0.050650, 0.476390, -0.973950, 0.040310, 0.007740, -0.203500, 
-0.338640, -0.565900, 0.159390, 0.305180, -0.017540, 0.167740, 0.279130, 
0.631640, 0.982850, -0.521180, 2.578970, -4.590810, -4.252250, -1.762120, 1.661280, 1.662510, 
-4.150710, -1.836320, -1.910410, -1.393830, 2.454280, -2.956780, -3.288440, -1.637310, -3.555420, 
2.778610, 2.656590, 1.754800, 1.336760, 4.330760, 3.364450, 2.315990, 0.654100, 1.070260, 

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
      0.0, -1.901780, 57,
       &Sources[0] , 
       &Weights[0] , 
      },
    { /* unit 59 (Old: 59) */
      0.0, 1.899260, 57,
       &Sources[57] , 
       &Weights[57] , 
      },
    { /* unit 60 (Old: 60) */
      0.0, 1.954300, 57,
       &Sources[114] , 
       &Weights[114] , 
      },
    { /* unit 61 (Old: 61) */
      0.0, 1.514620, 57,
       &Sources[171] , 
       &Weights[171] , 
      },
    { /* unit 62 (Old: 62) */
      0.0, -0.826790, 57,
       &Sources[228] , 
       &Weights[228] , 
      },
    { /* unit 63 (Old: 63) */
      0.0, -1.232440, 57,
       &Sources[285] , 
       &Weights[285] , 
      },
    { /* unit 64 (Old: 64) */
      0.0, -1.536360, 57,
       &Sources[342] , 
       &Weights[342] , 
      },
    { /* unit 65 (Old: 65) */
      0.0, 1.522140, 57,
       &Sources[399] , 
       &Weights[399] , 
      },
    { /* unit 66 (Old: 66) */
      0.0, -0.994350, 57,
       &Sources[456] , 
       &Weights[456] , 
      },
    { /* unit 67 (Old: 67) */
      0.0, -1.806480, 9,
       &Sources[513] , 
       &Weights[513] , 
      },
    { /* unit 68 (Old: 68) */
      0.0, 5.848550, 9,
       &Sources[522] , 
       &Weights[522] , 
      },
    { /* unit 69 (Old: 69) */
      0.0, -11.960650, 9,
       &Sources[531] , 
       &Weights[531] , 
      }

  };



int psinet2(float *in, float *out, int init)
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
