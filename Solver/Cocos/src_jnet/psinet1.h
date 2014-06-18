/*********************************************************
  psinet1.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:10:05 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int psinet1(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} psinet1REC = {340,3,psinet1};
