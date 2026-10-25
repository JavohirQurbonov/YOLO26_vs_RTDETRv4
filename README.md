# YOLO26 va RT-DETRv4: ketma-ket va parallel hisoblash

Maqola uchun tajriba kodlari va natijalari.

## Tarkibi
- common.py – umumiy qism (MiDaS, ketma-ket/parallel rejim, ovozli ogohlantirish)
- yolo26_app.py – YOLO26n dasturi
- rtdetrv4_app.py – RT-DETRv4-S dasturi
- results/ – kadr bo‘yicha o‘lchov natijalari (CSV)

## Ishga tushirish
pip install -r requirements.txt
python yolo26_app.py

RT-DETRv4 uchun: https://github.com/RT-DETRs/RT-DETRv4 repozitoriysini
yuklab oling, common.py va rtdetrv4_app.py ni uning ildiziga ko‘chiring,
vazn faylini (.pth) README dagi havoladan oling va python rtdetrv4_app.py ni ishga tushiring.

## Tajriba muhiti
Asus TUF Gaming, RTX 2050 4 GB, 16 GB RAM, Windows 11.

