from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .uploads import PhotoForm 
from . models import Photo
#PhotoFormクラスを使用する
# Create your views here.


def index(request):
    template = loader.get_template('clothes/index.html')  #loaderモジュールのメソッド
    content = {'form':PhotoForm()}
    #return render(request,template,content)
    return HttpResponse(template.render(content, request))


def predict(request):
    if not request.method == 'POST':
        return
        redirect('clothes:index')

    form = PhotoForm(request.POST,request.FILES)
    if not form.is_valid():
        raise ValueError('Formが不正です')

    photo = Photo(image=form.cleaned_data['image'])

    return HttpResponse()