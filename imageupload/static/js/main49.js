var filter = "win16|win32|win64|mac|macintel";


function readURL(input){
    if(input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
            $('#image_section').css('display', 'block');
            $('#image_section').attr('src', e.target.result);
            $('#image_section').css('height', '600px');
            $('#image_section').css('width', 'auto');
            $('#image_section').css('margin-top', '100px');
        }
        reader.readAsDataURL(input.files[0]);
    }
}

$(function() {
    $('#show_type').click(function() {
        if($('#image_section').css('display')=='block'){
            $('#jb-wide-contents-type').css('display', 'block');
            $('#pc-type-result').css('display', 'block');
        }
        }
    }
}
