package jafatt;

import java.util.ArrayList;
import java.util.StringTokenizer;
import org.biojava.bio.structure.AminoAcid;

public class Utilities{
    
    /* Get the separator of this system */
    public static String getSeparator(){
        String separator;
        
        /* Check the system */
        if(!System.getProperty("file.separator").equals("\\"))
            separator = "/";
        else
            separator = "\\";
        
        /* Return */
        return separator;
    }//getSeparator
    
    /* Get the relative postion of a given AA.
     It works like a parser on a string like
     AminoAcid ATOM:ILE I 264 true ATOM atoms: 21 */
    public static int getAAPosition(AminoAcid AA){
        String infoAA = AA.toString();
        String numStrAA;
        int numAA = -1;
        
        /* Tokes to leave out with the parser */
        StringTokenizer tkNum = new StringTokenizer(infoAA, " ");
        for(int i = 0; i < 3; i++)
            numStrAA = tkNum.nextToken();
        numStrAA = tkNum.nextToken();
        
        try{
            numAA = Integer.parseInt(numStrAA);
        }catch(NumberFormatException nfe){}
        
        return numAA;
    }//getAAPosition
    
    /* Return the string that selects a specific fragment */
    public static String selectFragmetString(Fragment frg){
        String cmd;
        /* More accurate version with check on selection (if based on atoms
         * or on AA):
         * if( frg.getParameters()[SATOM].equals(EMPTY) ||
           frg.getParameters()[EATOM].equals(EMPTY) ) 
         * ...
         * Deprecated 
         */
        /* cmd = "select (*)[" + frg.getParameters()[SATOM] + "][" 
                + frg.getParameters()[EATOM] + "]; "; */
        
        cmd = "select resno >=" + frg.getParameters()[Defs.FRAGMENT_START_AA] +
                    " and resno <=" + frg.getParameters()[Defs.FRAGMENT_END_AA];
        /* Debug */
        /* System.out.println("Command: " + cmd); */
        
        /* Return */
        return cmd;
    }//selectFragment
    
    public static String selectFragmetString(ArrayList<Fragment> selectedFragmentsA){
        String cmd;
        Fragment frg;
        int z = selectedFragmentsA.size();
        cmd = "select ";
        for(int i = 0; i < z; i++){
            frg = selectedFragmentsA.get(i);
            cmd = cmd + frg.getParameters()[Defs.FRAGMENT_START_AA] + "-"
                    + frg.getParameters()[Defs.FRAGMENT_END_AA];
            if(z == i+1)
                break;
            cmd = cmd + " or ";
            
        }
        cmd = cmd + ";";
        /* Debug */
        System.out.println("Command: " + cmd);
        
        return cmd;
    }//selectFragment
    
    /* Return the string that selects a specific atom */
    public static String selectAtomString(String atom, String numFrg){
        String cmd;
        cmd = "select (atomno = " + atom + " and */" + numFrg + "); ";
        return cmd;
    }//selectAtomString
    
    /* Return the string that selects a specific AA */
    public static String selectResidueString(String aa, String numFrg){
        String cmd;
        cmd = "select (resno = " + aa + " and */" + numFrg + "); ";
        return cmd;
    }//selectAtomString
    
    /* Return the string that prepare the Jmol panel for Assembling panel */
    public static String prepareStructureString(int numStructure){
        String cmd = "select */" + numStructure + "; ";
        
        //cmd = cmd + Defs.COMMAND_PREPARE_STRUCURE_BASE 
        //          + "select */" + numStructure + "; background hover red; "
        //          + "background labels black; axes molecular; axes on; ";
        cmd = cmd + Defs.COMMAND_PREPARE_STRUCURE_BASE 
                  + "select */" + numStructure + "; background hover red; "
                  + "background labels black; set allowMoveAtoms TRUE;"
                  + "set allowModelKit TRUE; set dynamicMeasurements ON; "
                  + "set allowRotateSelected TRUE; ";
        
        /* Display all the fragments */
            //cmd = cmd + "frame all; select *; center; ";
        /* Return string */
        return cmd;
    }//prepareStructureString
    
    public static String prepareStructureString(){
        String cmd;
        //cmd = cmd + Defs.COMMAND_PREPARE_STRUCURE_BASE 
        //          + "select */" + numStructure + "; background hover red; "
        //          + "background labels black; axes molecular; axes on; ";
        cmd = Defs.COMMAND_PREPARE_STRUCURE_BASE 
                + " background hover red; background labels black;"
                + " set allowMoveAtoms TRUE; set allowModelKit TRUE;"
                + " set dynamicMeasurements ON; set allowRotateSelected TRUE; "
                + " axes on; center; ";

        /* Display all the fragments */
            //cmd = cmd + "frame all; select *; center; ";
        /* Return string */
        return cmd;
    }//prepareStructureString
    
    /* Infos for deleted fragment */
    public static String deleteFragmentInfo(Fragment frg){
        String str;
        str = "Fragment deleted from residue [" 
               + frg.getParameters()[Defs.FRAGMENT_START_AA] + ": " 
               + frg.getParameters()[Defs.FRAGMENT_START_AA_NAME]
               + "] to residue [" + frg.getParameters()[Defs.FRAGMENT_END_AA] 
               + ": " + frg.getParameters()[Defs.FRAGMENT_END_AA_NAME] + "] " 
               + "on Extraction panel" ;
        return str;
    }//deleteFragmentInfo
    
    /* Infos for added fragment */
    public static String addFragmentInfo(Fragment frg){
        String str;
        
        str = fragmentInfo(frg) + " added";
        
        return str;
    }//addFragmentInfo
    
    public static String selectFragmentInfo(Fragment frg){
        String str;
        
        str = fragmentInfo(frg) + " selected";
        
        return str;
    }//addFragmentInfo
    
     public static String deselectFragmentInfo(Fragment frg){
        String str;
        
        str = fragmentInfo(frg) + " deselected";
        
        return str;
    }//addFragmentInfo
    
    /* Infos for a fragment */
    public static String fragmentInfo(Fragment frg){
        String str;
        str = "Fragment N " + frg.getParameters()[Defs.FRAGMENT_NUM] 
               + " [" + frg.getParameters()[Defs.FRAGMENT_START_AA] + ": " 
               + frg.getParameters()[Defs.FRAGMENT_START_AA_NAME]
               + "] [" + frg.getParameters()[Defs.FRAGMENT_END_AA] + ": "
               + frg.getParameters()[Defs.FRAGMENT_END_AA_NAME] + "]";
        return str;
    }//fragmentInfo
    
    /* Prepare string for switch view (see ViewOptionsPanel) */
    public static String switchViewString(String viewOption){
        String newView = "select *; spacefill off; wireframe off; trace off; "
                + "ribbon off; strands off; meshribbon off; "
                + "backbone off; dots off; cartoon off; ";
        if(viewOption.equals(Defs.NORMAL))
            newView = newView + "select backbone; wireframe 50; ";
        else if(viewOption.equals(Defs.SPACEFILL))
            newView = newView + "spacefill 50; ";
        else if(viewOption.equals(Defs.DOTS))
            newView = newView + "spacefill off; dots; ";
        else if(viewOption.equals(Defs.DOTSWIREFRAME))
            newView = newView + "wireframe 100; dots; ";
        else if(viewOption.equals(Defs.BACKBONE))
            newView = newView + "backbone 100; ";
        else if(viewOption.equals(Defs.TRACE))
            newView = newView + "trace; ";
        else if(viewOption.equals(Defs.RIBBON))
            newView = newView + "ribbon; ";
        else if(viewOption.equals(Defs.CARTOON))
            newView = newView + "cartoon; ";
        else if(viewOption.equals(Defs.STRANDS))
            newView = newView + "strands; ";
        else if(viewOption.equals(Defs.BALLANDSTICK))
            newView = newView + "wireframe 30; spacefill 75; ";
        else if(viewOption.equals(Defs.MESHRIBBONS))
            newView = newView + "meshribbon 2.5; ";
        return newView + "color structure; select *.CA; color red; select *; ";
    }//switchViewString
    
    /*
    * Input:
    * s1 = Protein on Extraction panel
    * s2 = Target sequence
    * Output:
    * common subsequence, start - end position of common subsequence in s1
    * NB: algoritmo cubico ma le stringhe sono corte!
    */
    public static String[] maxSubString(String s1, String s2){
        String max = "";
        int s = 0, e = 0;
        for (int i = 0; i < s1.length(); i++){
            
            /* i is the index of the substring s1 */
            for (int j = 0; j < s2.length(); j++){
                
                /* j is the index of the substring s2, k the length of the
                 substring */
                int k = 0;
                for (k = 0; i+k < s1.length() && j+k < s2.length(); k++)
                    if (s1.charAt(i+k) != s2.charAt(j+k))
                        
                        /* The common substring is over */
                        break;
                if (k > max.length()){
                    max = s1.substring(i, i+k);
                    s = i + 1; e = i + k + 1;
                }
            }
        }
        
        /* Return output */
        String output[] = new String[3];
        output[ 0 ] = max;
        output[ 1 ] = "" + s;
        output[ 2 ] = "" + e;
        return output;
    }//maxSubString

    /*public static int getResidueScaled(String sequence, String startRes, String endRes, int diff){
        for(int i = 0; i < sequence.length(); i++){
            if(sequence.charAt(i) == cv(startRes)){
                if((i + diff) < sequence.length())
                    if(sequence.charAt(i + diff) == cv(endRes))
                         return i;
            }
        }
        return 0;
    }//getResidueScaled*/

    /* Conversion for AA rep. to symbol */
    public static char cv(String AA){
        char a1 = ' ';
        if ("ALA".equals(AA)) a1= 'A';
        if ("ARG".equals(AA)) a1= 'R';
        if ("ASN".equals(AA)) a1= 'N';
        if ("ASP".equals(AA)) a1= 'D';
        if ("CYS".equals(AA)) a1= 'C';
        if ("GLN".equals(AA)) a1= 'Q';
        if ("GLU".equals(AA)) a1= 'E';
        if ("GLY".equals(AA)) a1= 'G';
        if ("HIS".equals(AA)) a1= 'H';
        if ("ILE".equals(AA)) a1= 'I';
        if ("LEU".equals(AA)) a1= 'L';
        if ("LYS".equals(AA)) a1= 'K';
        if ("MET".equals(AA)) a1= 'M';
        if ("PHE".equals(AA)) a1= 'F';
        if ("PRO".equals(AA)) a1= 'P';
        if ("SER".equals(AA)) a1= 'S';
        if ("THR".equals(AA)) a1= 'T';
        if ("TRP".equals(AA)) a1= 'W';
        if ("TYR".equals(AA)) a1= 'Y';
        if ("VAL".equals(AA)) a1= 'V';
        return a1;
    }//cv
    
    /* Replace a substring with another one with respect to the input offset */
    public static String replaceSubstring(String base, String subString, int offset){
        String outputString;
        char[] baseChar;
        char[] subStringChar;
        int baseLength;
        int subStringLength;
        int howMuchReplaced;
        
        /* Initialize elements */
        outputString = "";
        baseChar = base.toCharArray();
        subStringChar = subString.toCharArray();
        baseLength = base.length();
        subStringLength = subString.length();
        howMuchReplaced = 0;
        
        /* Strings start at position 0! */
        //offset = offset - 1;
        
        /* Check the offset and, in case, replace with the substring's chars */
        for(int i = 0; i < baseLength; i++){
            if(i >= offset && howMuchReplaced < subStringLength){
                outputString = outputString + subStringChar[howMuchReplaced];
                howMuchReplaced++;
            }else
               outputString = outputString + baseChar[i]; 
        }
        /* Return output */
        return outputString;
    }//replaceSubstring
    
    /* Get the string for connecting two fragments */
    public static String drawConnectionString(Fragment from, Fragment to){
        String cmd;
        int fromSAA;
        int fromEAA;
        int toSAA;
        int toEAA;
        int firstAA;
        int lastAA;
        
        try{
            fromSAA = Integer.parseInt(from.getParameters()[Defs.FRAGMENT_START_AA]);
            fromEAA = Integer.parseInt(from.getParameters()[Defs.FRAGMENT_END_AA]);
            toSAA = Integer.parseInt(to.getParameters()[Defs.FRAGMENT_START_AA]);
            toEAA = Integer.parseInt(to.getParameters()[Defs.FRAGMENT_END_AA]);
        }catch(NumberFormatException nfe){
            System.out.println("Error in parsing utilities: " + nfe);
            return "";
        }
        
        /* Check where to put the hook */
        if((fromEAA < fromSAA && fromSAA < toEAA) ||
           (toEAA < fromSAA && fromSAA < fromEAA))
            firstAA = fromSAA;
        else
            firstAA = fromEAA;
        
        if((fromEAA < toSAA && toSAA < toEAA) ||
           (toEAA < toSAA && toSAA < fromEAA))
            lastAA = toSAA;
        else
            lastAA = toEAA;
        
        /* Debug */
        /* System.out.println("fromSAA: " + fromSAA + " fromEAA: " + fromEAA
                + " toSAA: " + toSAA + " toEAA: " + toEAA 
                + " firstAA: " + firstAA + " lastAA: " + lastAA); */
        
         // draw connectLine 11  LINE (60.CA and */ 11) (91.CA and */ 18) COLOR red DIAMETER 4 TRANSLUCENT 
        
        /* Prepare command */
        cmd = "draw connectLine" + from.getParameters()[Defs.FRAGMENT_NUM]
              + " LINE (" + firstAA
              + ".CA and */" + from.getParameters()[Defs.FRAGMENT_NUM]
              + ") (" + lastAA
              + ".CA and */" + to.getParameters()[Defs.FRAGMENT_NUM]
              + ") COLOR red DIAMETER 4 TRANSLUCENT; ";
        
        /* Print connection line*/
        //System.out.println(cmd);
        
        /* Return command line */
        return cmd;
    }//drawConnectionString
    
    public static String deleteConnectionString(int frgsA){
        String cmd = "";
        for(int i=1;i<frgsA;i++){
            cmd = cmd + "delete $ connectLine" + i + " ; ";
        }
        return cmd;
    }
    
    /* Infos about connection */
    public static String connectionInfoString(Fragment from, Fragment to){
        return "AA[" + from.getParameters()[Defs.FRAGMENT_END_AA]
                     + " : " + from.getParameters()[Defs.FRAGMENT_END_AA_NAME]
                     + "] connected to AA["
                     + to.getParameters()[Defs.FRAGMENT_START_AA]
                     +" : " + to.getParameters()[Defs.FRAGMENT_START_AA_NAME]
                     + "]";
    }//connectionInfo
    
    /* Infos about constraint to set */
    public static String setConstraintString(int constraint, Fragment fragment){
        String info;
        
        info = "Set the constraint: ";
        switch(constraint){
            case Defs.CONSTRAINT_BLOCK:
                info = info + "BLOCK (BLOCK_)\n";
                info = info + "on fragment " + fragment.getParameters()[Defs.FRAGMENT_NUM];
                break;
            case Defs.CONSTRAINT_VOLUME:
                break;
            case Defs.CONSTRAINT_EXACTD:
                break;
            case Defs.CONSTRAINT_WITHIND:
                break;
            case Defs.CONSTRAINT_THESE_COORDINATES:
                break;
        }
        
        /* Return info string */
        return info;
    }//setConstraintString
    
    /* Get the color for the groups */
    public static String getGroupColor(String grp){
        String[] colors = Defs.GROUP_COLORS;
        int group;
        
        try{
            group = Integer.parseInt(grp);
        }catch(NumberFormatException nfe){
            return "";
        }
        
        /* Return the proper color */
        return colors[--group];
        
    }//getGroupColor
    
    
    
}//Utilities
