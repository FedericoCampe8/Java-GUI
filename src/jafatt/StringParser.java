package jafatt;

import java.util.StringTokenizer;

public class StringParser {
    
    /* Prefix of the output from Jmol*/
    private final static String MATCHPICK = "setStatusAtomPicked";
    private final static String MATCHMEASURE = "measurement";
    
    /*parse string str and return both atom and residure picked or measurement*/
    public static String[] parse(String str){
        /* Two kinds of outputs:
         * [model][Res_Name][Res_num][Atom][Ax][Ay][Az]
         * [MEASURE][numMis][Val][a][b][c][d]
         */
        String[] outStr = new String[7];
        String aux;
        int numModel;
        double coordinates;
        
        /* Tokes to leave out with the parsers */
        StringTokenizer tkRes = new StringTokenizer(str, "[");
        StringTokenizer tkCoordinates = new StringTokenizer(str, " )");
        StringTokenizer tkNumMeasure = new StringTokenizer(str, "[]");
        StringTokenizer tkNumRes = new StringTokenizer(str, "]:");
        StringTokenizer tkNumAtom = new StringTokenizer(str, "# ");
        StringTokenizer tkNumModel = new StringTokenizer(str, "./");
        
        /* Initalize the output string */
        for(int i = 0; i < 7; i++)
            outStr[ i ] = "";
        
        /* Pick an atom in the displayed structure */
        if(str.startsWith(MATCHPICK)){
            
            /* Parsing model's number */
            numModel = 0;
            while(tkNumModel.hasMoreTokens()){
                aux = tkNumModel.nextToken();
                if((aux.length() == 1) || (aux.length() == 2)){
                    try{
                        /* Selected model's number */
                        numModel = Integer.parseInt(aux); 
                    }catch(NumberFormatException nfe){
                        
                        /* Debug */
                        /* System.out.println("No model's number (1): " + aux); */
                        try{
                            aux = tkNumModel.nextToken();
                            numModel = Integer.parseInt(aux);
                        }catch(NumberFormatException nfe2){
                            System.out.println("No model's number (2): " + aux);
                        }
                    }
                    if(numModel != 0)
                        outStr[0] = aux;
                    numModel = 0;
                }
            }
            if (outStr[0].equals("")) 
                outStr[0] = "1";
            
            /* Parsing residue's name */
            tkRes.nextToken();
            aux = (tkRes.nextToken()).substring(0, 3);
            outStr[1] = aux;
            
            /* Parsing residue's number */
            tkNumRes.nextToken();
            aux =  tkNumRes.nextToken();
            outStr[2] = aux;
            
            /* Parsing atom's number */
            tkNumAtom.nextToken();
            aux = tkNumAtom.nextToken();
            outStr[3] = aux;
            
            /* Parsing atom's coordinates */
            coordinates = 0;
            tkCoordinates.nextToken();
            tkCoordinates.nextToken();
             for(int i = 0; i < 3; i++){
                aux = tkCoordinates.nextToken();
                try{
                    coordinates = Double.parseDouble(aux);
                }catch(NumberFormatException nfe){
                    System.out.println("Error in parsing atom's coordinates");
                }
                if(coordinates != 0){
                    outStr[4 + i] = aux;
                    coordinates = 0;
                } else
                    outStr[4 + i] = "";
                }
             
             /* Return infos about residue and atom picked */
             return outStr;
             
        /* Make a measurement in the displayed structure */
        }else if(str.startsWith(MATCHMEASURE)){
            StringTokenizer tkPrep = new StringTokenizer(str, "=");
            tkPrep.nextToken();
            String prep = tkPrep.nextToken();
            int l = prep.length();
            String prepM = " " + prep.substring(2, l - 1);
            StringTokenizer tkElementsMeasure = new StringTokenizer(prepM, ",");
            
            outStr[ 0 ] = "MEASURE";
            tkNumMeasure.nextToken();
            outStr[ 1 ] = tkNumMeasure.nextToken();
            
            /* Parsing the measurements */
            int i = 2;
            while(tkElementsMeasure.hasMoreTokens()){
                outStr[ i ] = tkElementsMeasure.nextToken();
                i++;
            }
            String[] outStrAux = new String[7];
            for(int j = 0; j < 7; j++) outStrAux[ j ] = "";
            outStrAux[ 0 ] = outStr[ 0 ];
            outStrAux[ 1 ] = outStr[ 1 ];
            outStrAux[ 2 ] = outStr[ i - 1 ];
            if(i-1 > 2) outStrAux[ 3 ] = outStr[ 2 ];
            if(i-1 > 3) outStrAux[ 4 ] = outStr[ 3 ];
            if(i-1 > 4) outStrAux[ 5 ] = outStr[ 4 ];
            if(i-1 > 5) outStrAux[ 6 ] = outStr[ 5 ];
            
            /* Return measurements */
            return outStrAux;
            
        /* Whatever else */  
        } else
            
           /* Return empty string */
           return outStr;
    }//parse
    
}
