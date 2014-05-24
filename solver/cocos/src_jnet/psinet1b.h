/*********************************************************
  psinet1b.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:10:01 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int psinet1b(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} psinet1bREC = {340,3,psinet1b};
