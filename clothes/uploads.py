from django import forms

class PhotoForm(forms.Form):  #Formクラスの継承
    image = forms.ImageField(widget=forms.FileInput(attrs={'class':'custom'}))