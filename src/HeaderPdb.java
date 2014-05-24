package jafatt;

import java.io.*;
import java.util.Scanner;

public class HeaderPdb {
    
    private static String proteinId;
    private static String targetSequence;
    private static int firstAA,lastAA,lastAAH;
    
    public static void info(String proteinPath){
        String[] text;
        String target = "";
        proteinId = "Null";
        int aa = -1;
        firstAA = -1;
        lastAA = -1;
        lastAAH = -1; //possible HETATM atom

        try {
            Scanner scanner = new Scanner(new File(proteinPath));

            while (scanner.hasNextLine()) {
                text = scanner.nextLine().split("\\s+");
                if (text[0].equals("ATOM") && firstAA == -1) {
                    firstAA = Integer.parseInt(text[5]);
                }
                if (text[0].equals("ATOM")) {
                    lastAA = Integer.parseInt(text[5]);
                    if (aa == -1) {
                        aa = Integer.parseInt(text[5]);
                        target = target + parse(text[3]);
                    }
                    if (aa + 1 == Integer.parseInt(text[5])) {
                        aa = Integer.parseInt(text[5]);
                        target = target + parse(text[3]);
                    }
                }
                if (text[0].equals("HETATM")) {
                    lastAAH = Integer.parseInt(text[5]);
                }
            }
            scanner.close();
        } catch (IOException e) {
        }
        targetSequence = target;
        System.out.println(target);
        System.out.println(firstAA + " " + lastAA + " " + lastAAH);

    }
    
    public static void targetId(String targetPath){
        String text[];
        try {
            Scanner scanner = new Scanner(new File(targetPath));
            text = scanner.nextLine().split(">|:");
            proteinId = text[1];
            //System.out.println(text[1]);
            targetSequence = scanner.nextLine();
            //System.out.println(targetSequence);
            scanner.close();
        } catch (IOException e) {}
    }
    
    
    public static int getFirstAA(){
        return firstAA;
    }
    
    public static int getLastAA(){
        return lastAA;
    }
    
    public static int getLastAAH(){
        return lastAAH;
    }
    
    public static String getProteinId(){
        return proteinId;
    }
    
    public static String getTargetSequence(){
        return targetSequence;
    }
     
     public static String parse(String aminoAcid){
         
         AminoAcids aa = AminoAcids.valueOf(aminoAcid);
         
         switch(aa){
             case ALA : return "A";
             case ARG : return "R";
             case ASN : return "N";
             case ASP : return "D";
             case ASX : return "B";
             case CYS : return "C";
             case GLU : return "E";
             case GLN : return "Q";
             case GLX : return "Z";
             case GLY : return "G";
             case HIS : return "H";
             case ILE : return "I";
             case LEU : return "L";
             case LYS : return "K";
             case MET : return "M";
             case PHE : return "F";
             case PRO : return "P";
             case SER : return "S";
             case THR : return "T";
             case TRP : return "W";
             case TYR : return "Y";
             case VAL : return "V";                 
         }
         return "";
     }
     
     public enum AminoAcids {
         ALA, ARG, ASN, ASP,
         ASX, CYS, GLU, GLN,
         GLX, GLY, HIS, ILE,
         LEU, LYS, MET, PHE,
         PRO, SER, THR, TRP,
         TYR, VAL,
  }
}
