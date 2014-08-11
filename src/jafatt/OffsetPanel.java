package jafatt;

import java.awt.GridLayout;
import java.util.ArrayList;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class OffsetPanel extends JPanel{
    
    ArrayList<JTextField> offsetFields = new ArrayList<JTextField>();
    String[] offsets;
    int numFragments;
    
    public OffsetPanel(ArrayList<Fragment> fragments){
        setup(fragments);
    }
    
    /* Setup the layout */
    private void setup(ArrayList<Fragment> fragments){
        int startAA;
        int endAA;
        numFragments = fragments.size();
        offsets = new String[numFragments];
        
        /* Initialize offsets */
        for(int i = 0; i < numFragments; i++)
            offsets[i] = Defs.EMPTY;
        
        /* Set the layout */
        setLayout(new GridLayout(numFragments + 1, 2));
        
        JLabel labelInfo = new JLabel("Fragments:");
        add(labelInfo);
        add(new JLabel("Offset:"));
        
        for(int i = 0; i < numFragments; i++){
            try{
                startAA = Integer.parseInt(
                        fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA]
                        );
                endAA = Integer.parseInt(
                        fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA]
                        );
            }catch(NumberFormatException nfe){
                return;
            }
            
            /* Labels on panel */
            JLabel labelFragment = new JLabel("[" + startAA + " ," 
                    + endAA + "] - L. " + (endAA - startAA) + " ");
            
            /* Offset fields on panel */
            JTextField offsetField = new JTextField();
            
            /* Add elements */
            offsetFields.add(offsetField);
            add(labelFragment);
            add(offsetField);
        } 
    }//setup
    
    /* Check (and save) offset if they are correct */
    public boolean checkData(String targetSequence){
        boolean ok;
        String offsetString;
        int offsetNumber;
        int maxNumber = targetSequence.length();
        
        /* Check offsets */
        for(int i = 0; i < numFragments; i++){
           offsetString = offsetFields.get(i).getText();
           
           /* Convert number */
           try{
               offsetNumber = Integer.parseInt(offsetString);
           }catch(NumberFormatException nfe){
               ok = false;
               return ok;
           }
           
           /* Check the correctness of the number */
           if(offsetNumber < 0 || offsetNumber > maxNumber){
               ok = false;
               return ok;
           }
           
           offsets[i] = offsetString;
        }
        
        /* Check if offsets are all different */
        for(int i = 0; i < numFragments - 1; i++){
            for(int j = i + 1; j < numFragments; j++){
                if(offsets[i].equals(offsets[j])){
                    ok = false;
                    return ok;
                }
            }
        }
        
        /* All is gone ok */
        ok = true;
        return ok;
    }//checkData
    
    /* Return the offsets */
    public String[] getOffsets(){
        return offsets;
    }//getOffsets
    
}//OffsetPanel
