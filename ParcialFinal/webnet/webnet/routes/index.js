var express = require('express');
var router = express.Router();
var exec = require('child_process').exec;

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index.html');
});

router.post('/image', function(req, res, next) {
    
  var shortid = require('shortid');
  var id = shortid.generate();
  
   exec('mkdir ./user_images/'+id+";cp ./user_images/*.* ./user_images/"+id+"/", function(error, stdout, stderr) {  
                    //console.log('stdout: ' + stdout);
                    //console.log('stderr: ' + stderr);
        if (error !== null) {
            console.log('exec error: ' + error);
        }
                
  
    require("fs").writeFile("./user_images/"+id+"/image.png", req.body.image, 'base64', function(err) {
        // llamar script python, meter lo que se tiene en esa funcion , borrar imagenes
        var path="./user_images/"+id 

        exec('cd '+path+'; ./sh.sh', function(error, stdout, stderr) { // llamamos a al script bash que ajustara la img y llamara a python e imprimimos solo el resultado
                //console.log('stdout: ' + stdout);
                //console.log('stderr: ' + stderr);
                if (error !== null) {
                    console.log('exec error: ' + error);
                }          
                res.writeHead(200, { 'Content-Type': 'application/json' });     
                res.end(JSON.stringify(stdout)); // send result of program      
                if (err){
                    console.log(err);
                }
            
            exec('rm -r ./user_images/'+id, function(error, stdout, stderr) {  // borramos imagenes luego de mostrar resultado
                    //console.log('stdout: ' + stdout);
                    //console.log('stderr: ' + stderr);
                    if (error !== null) {
                        console.log('exec error: ' + error);
                    }
                });
        });  });    
    });    
});


router.get('/indexa', function(req, res, next) {
  res.render('indexa.html');
});

module.exports = router;
