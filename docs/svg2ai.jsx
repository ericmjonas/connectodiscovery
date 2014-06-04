#target "Illustrator-16"

/* 
must have a file named filenames.txt in the current directory
write to this from an external script

infile outfile


filenames cannot have spaces (sorry)

*/ 


var xmlFile = new File("filenames.txt"); 
xmlFile.open("r");   
var xmlStr = xmlFile.read();  
xmlFile.close(); 

var filenames = xmlStr.split("\n")
for(var i = 0; i < filenames.length; ++i) { 
    if (filenames[i].length > 0) { 
        tgts = filenames[i].split(" "); 
        
        var infile = new File(tgts[0]);
        var myDocument = app.open(infile); 
        
        //Create a new text frame and assign it to the variable "myTextFrame" 
        

        saveInFile = new File( tgts[1]); 
        myDocument.saveAs(saveInFile)
    }
}

// app.quit()
// App.quit triggers a segfault for some reason so we have the calling script
// send a unix signal GOD THIS IS GROSS


