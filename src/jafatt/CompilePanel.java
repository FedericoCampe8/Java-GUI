package jafatt;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JCheckBox;
import javax.swing.BorderFactory;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Container;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.Dimension;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;

public class CompilePanel extends JFrame implements Runnable {

    private UserFrame view;
    private JPanel panel, buttonPanel;
    private OpButton compileButton, cancelButton;
    private InfoPanel ip;
    private JCheckBox fiasco, cocos, cuda;
    private Boolean running = true;
    private int exitValue;
    private Process p;
    private Compiler c;
    private Thread tc;

    public CompilePanel(UserFrame view) {

        super("Fiasco - Compile Solvers");

        this.view = view;

        panel = new JPanel(new BorderLayout());
        buttonPanel = new JPanel();

        double w = view.getBounds().width;
        double h = view.getBounds().height;
        double widthFrame = (w * 40.0) / 100.0;  //960
        double heightFrame = (h * 30.0) / 100.0;  //540

        /* Setup layout */
        setPreferredSize(new Dimension((int) widthFrame, (int) heightFrame));
        setResizable(true);


        compileButton = new OpButton("Compile", "") {
            @Override
            public void buttonEvent(ActionEvent evt) {
                buttonEvent(true);
            }
        };

        cancelButton = new OpButton("Cancel", "") {
            @Override
            public void buttonEvent(ActionEvent evt) {
                buttonEvent(false);
            }
        };

        c = new Compiler();

        buttonPanel.add(compileButton);
        buttonPanel.add(cancelButton);

        panel.add(c, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);

        addWindowListener(new WindowAdapter() {

            @Override
            public void windowClosing(WindowEvent e) {
                running = false;
            }
        });

    }

    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(panel);
        update();
        setVisible(true);
    }//run

    private void update() {
        pack();
        setLocationRelativeTo(view);
    }

    private void buttonEvent(boolean ok) {

        if (ok) {
            cancelButton.setEnabled(false);
            buildScript();
            tc = new Thread(c);
            tc.start();
            //setVisible(false);
            //dispose();

        } else {
            setVisible(false);
            dispose();
        }
    }
    
    
    public void buildScript(){
        String script = "#!/bin/bash\n";
        if(cocos.isSelected()){
            script += "echo ------------------ COCOS ------------------\n" +
                    "cd ./Cocos;\n"+
                    "make clean;\n"+
                    "./CompileAndRun.sh\n"+
                    "cd ..\n";
        }
        if(fiasco.isSelected()){
            script += "echo ------------------ FIASCO ------------------\n"+
                    "cd ./Fiasco;\n"+
                    "make clean;\n"+
                    "make;\n"+
                    "cd ..\n";
        }
        if(cuda.isSelected()){
            script += "echo ------------------ COCOS GPU ------------------\n"+
                    "cd ./CocosGPU;\n"+
                    "make clean;\n"+
                    "./CompileAndRun.sh\n"+
                    "cd ..\n";
                    
        }
        try {
            FileOutputStream output = new FileOutputStream(
                    Defs.SOLVER_PATH + "/compile.sh");
            output.close();
        } catch (IOException e) {}
        try{
            BufferedWriter bw = new BufferedWriter(
                    new FileWriter(Defs.SOLVER_PATH + "/compile.sh", true));
            bw.write(script);
            bw.flush();
            bw.close();
        } catch (Exception e) {}
        
    }

    private class Compiler extends JPanel implements Runnable {

        private JPanel subPanel;

        public Compiler() {

            setLayout(new BorderLayout());
            subPanel = new JPanel(new FlowLayout());
            ip = new InfoPanel(false);
            cocos = new JCheckBox("Compile Cocos",true);
            cocos.setEnabled(true);
            fiasco = new JCheckBox("Compile Fiasco",true);
            fiasco.setEnabled(true);
            cuda = new JCheckBox("Compile CocosGPGPU");
            cuda.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    if(cuda.isSelected())
                        ip.printStringLn("Check your graphic card at https://developer.nvidia.com/cuda-gpus before"
                                + " compiling CocosGPGU");
                }
            });
            
            
            subPanel.add(cocos);
            subPanel.add(fiasco);
            subPanel.add(cuda);


            add(ip, BorderLayout.CENTER);
            add(subPanel, BorderLayout.SOUTH);
            setBorder(BorderFactory.createTitledBorder("Compile Cocos & Fiasco"));

        }

        @Override
        public void run() {

            try {
                Process permission = new ProcessBuilder("chmod", "+x", Defs.SOLVER_PATH + "compile.sh").start();
                permission.waitFor();

                String line = "";
                exitValue = -1;
                ProcessBuilder process = new ProcessBuilder("./compile.sh");
                process.directory(new File(Defs.SOLVER_PATH));
                p = process.start();
                BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
                try {
                    if (running) {
                        while ((line = reader.readLine()) != null) {
                            ip.printStringLn(line);
                        }
                    }
                } catch (IOException e) {
                }
                p.destroy();
                p.waitFor();
                exitValue = p.exitValue();
                running = false;
            } catch (Exception e) {
                e.printStackTrace(System.out);
            }
            if (exitValue != 0) {
                view.printStringLn("Error while Compiling");
                return;
            }else{
                cancelButton.setEnabled(true);
                cancelButton.setText("Exit");
                
            }

        }//run
    }
}