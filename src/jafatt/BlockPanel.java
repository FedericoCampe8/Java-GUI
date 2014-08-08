package jafatt;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;

public class BlockPanel extends JPanel{
    
    private String fragmentNum;
    private OperationalPanel opPanel;
    private JScrollPane scrollPane;
    
    /* Local copy, subject to modifies */
    private ArrayList<String[]> groupsInfo;
    
    private boolean setConstraint;
    private String[] fragmentBlockInfos;
    
    /* Groups Table */
    private JTable groupsTable;
    private int numCol;
    private String[] columnNames;
    private Object[][] data;
    
    public BlockPanel(Fragment fragment, ArrayList<String[]> groupsInfo){
        setup(fragment, groupsInfo);
    }
    
    /* Setup the panel */
    private void setup(Fragment fragment, ArrayList<String[]> groupsInfo){
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 35.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 40.0) / 100.0;
        
        /* Setup layout */
        setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));
        setLayout(new GridLayout(2, 1));
        
        fragmentNum = fragment.getParameters()[Defs.FRAGMENT_NUM];
        opPanel = new OperationalPanel(this, fragment);
        setConstraint = false;
        fragmentBlockInfos = new String[2];
        int size = groupsInfo.size();
        
        this.groupsInfo =  new ArrayList<String[]>(); 
        
        /* Make a deep copy (maybe it not necessary) */
        for(int i = 0; i < size; i++){
            String[] infos = groupsInfo.get(i);
            String[] infosCloned = new String[2];
            infosCloned[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM] = infos[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM];
            infosCloned[Defs.CONSTRAINT_BLOCK_GROUP] = infos[Defs.CONSTRAINT_BLOCK_GROUP];
            this.groupsInfo.add(infos);
        }
        
        /* Set the info table */
        setTable(size);
        scrollPane = new JScrollPane(groupsTable);
        
        /* Add elements */
        add(scrollPane, 0);
        add(opPanel, 1);
    }//setup
    
    /* Setup the group-info Table */
    private void setTable(int numRows){
        
        /* Columns */
        numCol = 3;
        columnNames = new String[numCol];
        columnNames[0] = "Group";
        columnNames[1] = "N_Fragment";
        columnNames[2] = "Group's_Color";
        
        /* Rows */
        data = new Object[numRows][numCol];
        
        /* Fill data */
        for(int i = 0; i < numRows; i++){
            String grp = groupsInfo.get(i)[Defs.CONSTRAINT_BLOCK_GROUP];
            String frgNum = groupsInfo.get(i)[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM];
            String color = Utilities.getGroupColor(grp);
            
            data[i][0] = grp;
            data[i][1] = frgNum;
            data[i][2] = color;
            
            /* Debug */
            /* System.out.println("GroupsInfo size: " + numRows + " grp: " 
                    + grp + " fragment Number: " + frgNum + " color: " + color); */
        }
        
        /* Create a new Table with the data */
        groupsTable = new JTable(data, columnNames);
        groupsTable.setColumnSelectionAllowed(false);
        groupsTable.setRowSelectionAllowed(false);
        groupsTable.setCellSelectionEnabled(false);
        groupsTable.setFillsViewportHeight(true);
        
        /* Add to scrollPane */
        scrollPane = new JScrollPane(groupsTable);
    }//setTable
    
    /* Verify whether to set the block constraint */
    public boolean constraintToSet(){
        return setConstraint;
    }//constraintToSet
    
    /* Update blockInfos */
    public String[] getBlockGroup(){
        return fragmentBlockInfos;
    }//updateBlockGroups
    
    private class OperationalPanel extends JPanel{
        
        private JPanel parentPanel;
        
        private JTextField addFragmentToGroup;
        private JLabel infoLabel;
        
        private JButton addGroupB;
        private JButton addFragmentGroupB;
        private JButton deleteFragmentGroupB;
        
        private Fragment fragment;
        
        private OperationalPanel(JPanel parentPanel, Fragment fragment){
            setup(parentPanel, fragment);
        }
        
        /* Setup */
        private void setup(JPanel parentPanel, Fragment fragment){
            this.fragment = fragment;
            this.parentPanel = parentPanel;
            
            setLayout(new GridLayout(4, 2));
            
            addFragmentToGroup = new JTextField();
            infoLabel = new JLabel();
            infoLabel.setText(" Fragment N " + fragmentNum + ": [" 
                    + fragment.getParameters()[Defs.FRAGMENT_START_AA] + ", "
                    + fragment.getParameters()[Defs.FRAGMENT_END_AA] + "]");
            
            addFragmentGroupB = new JButton("Add");
            addFragmentGroupB.addActionListener(new ActionListener(){
                @Override
                public void actionPerformed(ActionEvent evt){
                    addFragmentGroupAction();
                }
            });
            addGroupB = new JButton("Create");
            addGroupB.addActionListener(new ActionListener(){
                @Override
                public void actionPerformed(ActionEvent evt){
                    addGroupAction();
                }
            });
            deleteFragmentGroupB = new JButton("Delete");
            deleteFragmentGroupB.addActionListener(new ActionListener(){
                @Override
                public void actionPerformed(ActionEvent evt){
                    deleteAction();
                }
            });
            
            add(new JLabel("Add fragment N " + fragmentNum));
            add(infoLabel);
            add(addFragmentToGroup);
            add(addFragmentGroupB);
            add(new JLabel("Create a new group"));
            add(new JLabel("Delete fragment N " + fragmentNum + " from its group"));
            add(addGroupB);
            add(deleteFragmentGroupB);
        }//setup
        
        /* Add a new group */
        private void addGroupAction(){
            int maxGroup = findMaxGroup();
            
            if(maxGroup == Defs.GROUP_COLORS_NUM){
                infoLabel.setText("Reached the maximum number of groups");
                setConstraint = false;
                return;
            }
            
            maxGroup++;
            
            fragmentBlockInfos[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM] = "";
            fragmentBlockInfos[Defs.CONSTRAINT_BLOCK_GROUP] = "" + maxGroup;
            
            groupsInfo.add(fragmentBlockInfos);
            
            
            /* Change table */
            changeTable();
            
            /* Print info */
            infoLabel.setText("Group " + maxGroup + " added");
        }//addGroupAction
        
        /* Add a fragment on a specified group */
        private void addFragmentGroupAction(){
            int group;
            
            /* Read the group written on the Text field */
            String groupString = addFragmentToGroup.getText();
            
            if(groupString.isEmpty()){
                infoLabel.setText("Write a number first");
                setConstraint = false;
                return;
            }
            
            /* Check if it is a proper number */
            try{
                group = Integer.parseInt(groupString);
            }catch(NumberFormatException nfe){
                infoLabel.setText("Write a proper number");
                setConstraint = false;
                return;
            }
            
            if(group < 1 || group > findMaxGroup()){
                infoLabel.setText("Group's bounds exceeded");
                setConstraint = false;
                return;
            }
            
            if(fragmentAlreadyPresent()){
                infoLabel.setText("Fragment already present");
                setConstraint = false;
                return;
            }
            
            /* Delete the empty group */
            int size = groupsInfo.size();
            for(int i = 0; i < size; i++){
                if(groupsInfo.get(i)[Defs.CONSTRAINT_BLOCK_GROUP].equals(groupString)
                        &&
                   groupsInfo.get(i)[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM].isEmpty()){
                   groupsInfo.remove(i);
                   break;
                }
            }
            
            fragmentBlockInfos[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM] = 
                    fragment.getParameters()[Defs.FRAGMENT_NUM];
            fragmentBlockInfos[Defs.CONSTRAINT_BLOCK_GROUP] = groupString;
            
            groupsInfo.add(fragmentBlockInfos);
            
            /* Change table */
            changeTable();
            
            /* Print info */
            infoLabel.setText("Fragment added to group " + group);
            
            setConstraint = true; 
        }//addFragmentGroupAction
        
        /* Delete fragment from its group */
        private void deleteAction(){
            int size = groupsInfo.size();
            String fragmentNum = fragment.getParameters()[Defs.FRAGMENT_NUM];
            
            for(int i = 0; i < size; i++){
                if(fragmentNum.equals(
                        groupsInfo.get(i)[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM])){
                    groupsInfo.remove(i);
                    infoLabel.setText("Fragment's group removed");
                    break;
                }
            }
            
            /* Change table */
            changeTable();
            
            setConstraint = false;
        }//deleteAction
        
        /* Find the max number's group already set */
        private int findMaxGroup(){
            int maxGroup = 0;
            int numElements = groupsInfo.size();
            int groupNumber = 0;
            
            for(int i = 0; i < numElements; i++){
                try{
                    groupNumber = Integer.parseInt(
                            groupsInfo.get(i)[Defs.CONSTRAINT_BLOCK_GROUP]);
                }catch(NumberFormatException nfe){
                    System.out.println("Error in Block Panel: " + nfe);
                    setConstraint = false;
                    return 0;
                }
                
                if(maxGroup < groupNumber)
                    maxGroup = groupNumber;
            }
            
            /* Return the max group number (note the base case) */
            if(numElements == 1)
                maxGroup = groupNumber;
            return maxGroup;     
        }//findMaxGroup
        
        /* Check whether the fragment has already a set group */
        private boolean fragmentAlreadyPresent(){
            int size = groupsInfo.size();
            String fragmentNum = fragment.getParameters()[Defs.FRAGMENT_NUM];
            
            for(int i = 0; i < size; i++){
                String frgNum = groupsInfo.get(i)[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM];
                if(fragmentNum.equals(frgNum))
                    return true;
            }
            
            return false;
        }//fragmentAlreadyPresent
        
        /* Update the table */
        private void changeTable(){
            
            /* Debug */
            /* System.out.println("Change table"); */
            
            setTable(groupsInfo.size());
            parentPanel.remove(0);
            parentPanel.add(scrollPane, 0);
            parentPanel.validate();
        }//changeTable
        
    }//OperationalPanel
    
}//BlockPanel
