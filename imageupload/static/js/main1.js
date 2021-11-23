

var ex_file = document.getElementById("ex_file");

ex_file.addEventListener('click', function(e){
    console.log(e.target.files)
})

var file = e.target.files[0];
var reader = new FileReader();
reader.readAsDataURL(file);

reader.onload =