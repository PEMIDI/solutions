---
github: "true"
---


پیمان: 😕 بچه‌ها، من یه مشکل جدی پیدا کردم توی اپلیکیشن Uber ما. وقتی کاربرها لیست سفرهاشون رو لود می‌کنن، سیستم خیلی کند میشه. انگار داره همه اطلاعات رو یکجا می‌کشه بالا. نمی‌دونم چجوری باید حلش کنم.

حسین: 🤔 اوه، این می‌تونه مشکل بزرگی باشه پیمان جان. بذار ببینم دقیقاً مشکل کجاست. میشه یکم بیشتر توضیح بدی که چه اتفاقی میفته؟

پیمان: 😟 آره حتماً. ببین، وقتی کاربر وارد صفحه تاریخچه سفرهاش میشه، ما داریم تمام سفرهای قبلی رو با جزئیات کامل از دیتابیس می‌کشیم بیرون. این شامل اطلاعات راننده، مسیر، قیمت و حتی نظرات هم میشه. برای کاربرهایی که سفرهای زیادی دارن، این پروسه خیلی طول می‌کشه و اپ کند میشه.

حسین: 🧐 آهان، متوجه شدم. ببین پیمان جان، به نظرم ما اینجا می‌تونیم از تکنیک Lazy Loading در مقابل Eager Loading استفاده کنیم. [🔗](https://docs.microsoft.com/en-us/ef/core/querying/related-data/eager)

ماهان: 😅 ببخشید که می‌پرم وسط حرفتون، ولی میشه یکم بیشتر توضیح بدین؟ من تازه شروع کردم و این اصطلاحات برام جدیده.

حسین: 😊 البته ماهان جان، خوشحالم که می‌پرسی. ببین، Eager Loading یعنی ما همه داده‌های مرتبط رو یکجا و همون اول کار لود می‌کنیم. [🔗](https://www.entityframeworktutorial.net/eager-loading-in-entity-framework.aspx) این همون کاریه که الان داریم انجام میدیم و باعث کندی سیستم شده. اما Lazy Loading یعنی ما داده‌ها رو فقط وقتی که لازم داریم، لود می‌کنیم. 

پیمان: 🤨 خب این چطوری می‌تونه به ما کمک کنه؟

حسین: 👨‍🏫 ببینید بچه‌ها، ایده اینه که ما اول فقط اطلاعات اصلی سفرها رو لود کنیم، مثل تاریخ و مقصد. بعد وقتی کاربر روی یه سفر خاص کلیک کرد، اون موقع جزئیات بیشتر رو لود می‌کنیم. این باعث میشه که صفحه اصلی خیلی سریع‌تر لود بشه.

ماهان: 🙋‍♂️ میشه یه مثال کد بزنید؟ اینجوری بهتر متوجه میشم.

حسین: 😃 حتماً ماهان جان. ببین، الان کد ما احتمالاً چیزی شبیه این هست:

```python
def get_user_trips(user_id):
    return Trip.objects.filter(user_id=user_id).select_related('driver').prefetch_related('reviews')
```

این کد همه چیز رو یکجا میاره. اما می‌تونیم اینطوری تغییرش بدیم:

```python
def get_user_trips(user_id):
    return Trip.objects.filter(user_id=user_id).only('id', 'date', 'destination')

def get_trip_details(trip_id):
    return Trip.objects.select_related('driver').prefetch_related('reviews').get(id=trip_id)
```

حالا `get_user_trips` فقط اطلاعات اصلی رو میاره، و `get_trip_details` رو وقتی صدا می‌زنیم که کاربر جزئیات یه سفر خاص رو خواست. [🔗](https://docs.djangoproject.com/en/3.2/ref/models/querysets/#select-related)

پیمان: 🤯 وای، این عالیه! فکر کنم خیلی به بهبود عملکرد کمک کنه.

مارال: 👩‍💼 سلام بچه‌ها، ببخشید که دیر اومدم. حسین جان، این ایده‌ت خیلی خوبه. فقط یه سوال، این تغییر روی API ما هم تاثیر میذاره؟

حسین: 🤓 سلام مارال جان، سوال خوبیه. بله، احتمالاً باید API رو هم آپدیت کنیم. می‌تونیم دو اندپوینت جدا داشته باشیم:

1. `/api/trips/` برای لیست کلی سفرها
2. `/api/trips/<trip_id>/` برای جزئیات هر سفر

اینجوری کلاینت‌ها هم می‌تونن داده‌ها رو به صورت تدریجی دریافت کنن. [🔗](https://www.django-rest-framework.org/api-guide/viewsets/#readonlymodelviewset)

مارال: 👍 عالیه. پیمان جان، می‌تونی این تغییرات رو اعمال کنی و یه تست پرفورمنس بگیری؟ می‌خوام ببینم چقدر بهبود حاصل میشه.

پیمان: 😊 حتماً! من همین الان شروع می‌کنم. ممنون حسین جان بابت راهنمایی‌ت.

حسین: 🙌 خواهش می‌کنم. یادتون باشه، گاهی اوقات بهینه‌سازی یعنی انجام ندادن کارها تا وقتی که واقعاً لازم بشن. Lazy Loading دقیقاً همین کار رو می‌کنه. [🔗](https://en.wikipedia.org/wiki/Lazy_loading)

ماهان: 🤓 من خیلی یاد گرفتم. مرسی که اینقدر خوب توضیح دادید.

مارال: 😄 خیلی خوبه که تیم اینقدر خوب با هم کار می‌کنه. حسین جان، لطفاً یه داکیومنت هم در مورد این تغییر بنویس تا بقیه تیم هم در جریان باشن.