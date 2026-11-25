"""Live OpenAI + FastAPI regression, with two concurrent chats sharing isolated state."""
import asyncio, json, os, shutil, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
OUT = Path(tempfile.mkdtemp(prefix='flowers-two-chat-api-'))
STATE = OUT / 'state'; STATE.mkdir()
for filename in ('prices.json', 'removed.json', 'overrides.json', 'custom.json'):
    source = ROOT / filename
    if source.exists(): shutil.copy2(source, STATE / filename)
ADMIN, A, B = '905550009999', '905550000101', '905550000102'
for key, value in dict(STATE_DIR=str(STATE), ADMIN_NUMBERS=ADMIN, ADMIN_TELEGRAM='', GRAPH_PATH=str(ROOT/'data/graph.json'),
    DEBOUNCE_SECONDS='0.05', NOTIFY_ASYNC='0', RESEND_CHECK='0', LATE_CHECK='0', CHARACTER_AUTO='0', DL_PRELOAD='0',
    SANDBOX_PRELOAD='0', WA_CHECK='0', RATE_MESSAGES='1000', META_TEST_MODE='0', WAHA_API_KEY='', WAHA_WEBHOOK_SECRET='',
    WHATSAPP_TOKEN='', PHONE_NUMBER_ID='', TELEGRAM_TOKEN='', TELEGRAM_SECRET='', PUBLIC_URL='', INTERPRET='1').items():
    os.environ[key] = value
assert os.environ.get('OPENAI_API_KEY'), 'OPENAI_API_KEY is missing'
import httpx, agent, core
from app import app, is_mock
assert is_mock()
core.TRANSPORT = None; core.TG = None
TRANSCRIPT, CHECKS = [], []

def checkpoint(name, condition, detail=''):
    CHECKS.append({'check': name, 'passed': bool(condition), 'detail': detail})
    print(('PASS ' if condition else 'FAIL ') + name + (' ' + str(detail) if detail else ''), flush=True)
    save()
    assert condition, name + ': ' + str(detail)

def save():
    (OUT/'report.json').write_text(json.dumps({'checks': CHECKS, 'transcript': TRANSCRIPT, 'orders': core.orders()}, ensure_ascii=False, indent=2))

def latest(phone):
    return next((r for r in reversed(core.orders()) if r['customer_number'] == phone), None)

async def send(client, phone, text):
    start = len(core.FEED); t = time.monotonic()
    TRANSCRIPT.append({'from': phone, 'text': text})
    response = await client.post('/message', data={'from': phone, 'name': 'Owner' if phone == ADMIN else 'Ava Test' if phone == A else 'Deniz Test', 'text': text})
    response.raise_for_status()
    while agent.DEB.t:
        jobs = list(agent.DEB.t.values())
        await asyncio.gather(*jobs, return_exceptions=False)
        if all(job.done() for job in agent.DEB.t.values()): break
    events = [dict(e) for e in core.FEED[start:] if e.get('kind') in ('text','document','image','log')]
    TRANSCRIPT.append({'request_from': phone, 'elapsed_s': round(time.monotonic()-t,2), 'events': events})
    save()
    print('TURN', phone, text[:65], round(time.monotonic()-t,1), flush=True)
    return events

async def pair(client, ta, tb):
    return await asyncio.gather(send(client,A,ta),send(client,B,tb))

async def main():
    agent.LOOP = asyncio.get_running_loop()
    agent.DEB = agent.Debouncer(agent.process_batch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',timeout=180) as client:
        print('ARTIFACTS', OUT, flush=True)
        await pair(client,
            'I want 1 x 5 White Roses https://kibrisciceksepetim.com/tr/products/5-beyaz-gul . Card message: love u baby. That is all.',
            '1 adet 140 cm peluş ayı istiyorum https://kibrisciceksepetim.com/tr/products/140-cm-pelus-ayi . Kart notu: seni seviyorum. Hepsi bu.')
        checkpoint('two distinct baskets created', latest(A) is not None and latest(B) is not None and latest(A)['items'] != latest(B)['items'])
        await pair(client,
            'Your name: Ava Test\nYour phone: +905550000101\nRecipient name: Ava Test\nRecipient phone: +905550000101\nCity: Lefkosa\nAddress: Lefkosa, Test Street, No 42, apartment 7, opposite the cafe',
            'Gönderen adı: Deniz Test\nGönderen telefonu: +905550000102\nAlıcı adı: Deniz Test\nAlıcı telefonu: +905550000102\nŞehir: Lefkoşa\nAdres: Lefkoşa, Deneme Sokak, No 24, Daire 3, kafenin karşısı')
        # The site may require one explicit basket acknowledgement before the delivery menu.
        for phone in (A,B):
            row = latest(phone)
            if core.kget('bok:'+row['order_id']) != '|'.join(str(row[k]) for k in ('items','quantities','custom_items','custom_price')):
                await send(client,phone,'yes, that is all' if phone==A else 'evet, hepsi bu')
        checkpoint('both personal forms saved', not core.form_missing(latest(A)) and not core.form_missing(latest(B)), str({p:core.form_missing(latest(p)) for p in (A,B)}))
        checkpoint('delivery menus sent to both', all(core.kget('dopts:'+latest(p)['order_id']) for p in (A,B)))
        baskets = {p:(latest(p)['items'],latest(p)['quantities']) for p in (A,B)}
        await pair(client,'1, tomorrow','2, yarın 15:00')
        checkpoint('numeric delivery replies preserve both baskets', all((latest(p)['items'],latest(p)['quantities'])==baskets[p] for p in (A,B)))
        checkpoint('normal and jet saved independently', latest(A)['delivery_type']=='normal' and latest(B)['delivery_type']=='jet' and latest(B)['delivery_time']=='15:00')
        await pair(client,'yes all good, submit it','evet hepsi doğru, siparişi gönder')
        ids = {p:latest(p)['order_id'] for p in (A,B)}
        checkpoint('both orders submitted', all(core.kget('subm:'+ids[p]) for p in (A,B)))
        await send(client,ADMIN,'/turkish')
        await asyncio.gather(send(client,ADMIN,'/confirm #'+ids[A]),send(client,ADMIN,'/confirm +'+B))
        checkpoint('slash confirms target correct two orders', all(latest(p)['status']=='awaiting_payment' for p in (A,B)))
        start = len(core.FEED)
        await asyncio.gather(send(client,ADMIN,'/reject #'+ids[A]+' we are closed tomorrow for holiday'),send(client,ADMIN,'/reject +'+B+' peluş ayı stokta yok'))
        notices = [e for e in core.FEED[start:] if e.get('to') in (A,B) and e.get('kind')=='text']
        checkpoint('both confirmed orders rejected', all(latest(p)['status']=='rejected' for p in (A,B)))
        checkpoint('rejection sends no product images', not any(e.get('kind')=='image' and e.get('to') in (A,B) for e in core.FEED[start:]))
        checkpoint('English customer hears holiday reason in English', any(e['to']==A and 'holiday' in e['body'].lower() for e in notices), '\n'.join(e['body'] for e in notices if e['to']==A))
        checkpoint('Turkish customer hears unavailable item and opt-in', any(e['to']==B and ('stok' in e['body'].lower() or 'mevcut' in e['body'].lower()) and '?' in e['body'] for e in notices), '\n'.join(e['body'] for e in notices if e['to']==B))
        await asyncio.gather(send(client,ADMIN,'/confirm +'+A),send(client,ADMIN,'/confirm #'+ids[B]))
        checkpoint('both rejection decisions reopened by admin', all(latest(p)['status']=='awaiting_payment' for p in (A,B)))
        checkpoint('reopening keeps original order IDs', all(latest(p)['order_id']==ids[p] for p in (A,B)))
        await send(client,ADMIN,'reject order #'+ids[A]+' because we are closed tomorrow')
        checkpoint('natural rejection targets A only',latest(A)['status']=='rejected' and latest(B)['status']=='awaiting_payment')
        await send(client,ADMIN,'nevermind, confirm it')
        checkpoint('natural reversal keeps focused rejected A',latest(A)['status']=='awaiting_payment' and latest(B)['status']=='awaiting_payment')
        # Inject an actual persistence failure for A only, while B continues normally.
        original = core.request_change
        def broken(oid,phone,*args,**kwargs):
            if phone==A: raise OSError('SIMULATED disk failure for customer A')
            return original(oid,phone,*args,**kwargs)
        core.request_change=broken
        try:
            start=len(core.FEED)
            await pair(client,'Please change my delivery address to Lefkosa, New Test Street 99, apartment 4, opposite the school.','Siparişimin kart notunu iyi ki varsın olarak değiştirir misiniz?')
        finally:
            core.request_change=original
        checkpoint('failed save immediately alerts admin',any(e.get('to')==ADMIN and 'request_change' in e.get('body','') and ('kaydet' in e.get('body','').lower() or 'SAVING FAILED' in e.get('body','')) for e in core.FEED[start:]))
        checkpoint('only failed chat is paused',bool(core.q('SELECT 1 FROM paused WHERE phone=?',A)) and not core.q('SELECT 1 FROM paused WHERE phone=?',B))
        count=len([e for e in core.FEED if e.get('to')==A and e.get('kind')=='text'])
        await send(client,A,'Should I send the address again?')
        checkpoint('follow-up does not reask after save failure',count==len([e for e in core.FEED if e.get('to')==A and e.get('kind')=='text']))
        await send(client,ADMIN,'/unmute +'+A)
        start=len(core.FEED)
        await send(client,ADMIN,'Tell the customer of order #'+ids[A]+': Teslimat için ekibimiz sizinle iletişime geçecek.')
        forwarded=[e['body'] for e in core.FEED[start:] if e.get('to')==A and e.get('kind')=='text']
        checkpoint('Turkish admin forwards message in English customer language',bool(forwarded) and any('delivery' in text.lower() for text in forwarded) and not any('teslimat' in text.lower() for text in forwarded), str(forwarded))
        await asyncio.gather(send(client,ADMIN,'/reject #'+ids[A]),send(client,ADMIN,'/reject #'+ids[B]))
        checkpoint('slash rejection cancels order even with pending change',all(core.order_row(ids[p])['status']=='rejected' for p in (A,B)))
        await pair(client,
            'A separate NEW order please: 1 x Bouquet of 11 Roses https://kibrisciceksepetim.com/tr/products/11-li-gul-buketi . Sender Ava Newname, phone +905550000111. Recipient Zoe Newname, phone +905550000112. Use the same address as my previous order. Normal delivery tomorrow. New card message: Happy birthday. That is all.',
            'Ayrı YENİ sipariş istiyorum: 1 adet 5 Beyaz Gül https://kibrisciceksepetim.com/tr/products/5-beyaz-gul . Gönderen Deniz Yeni, telefon +905550000121. Alıcı Ece Yeni, telefon +905550000122. Önceki siparişimdeki aynı adrese, yarın normal teslimat. Yeni kart notu: iyi ki doğdun. Hepsi bu.')
        checkpoint('cancelled customers create new order IDs',all(latest(p)['order_id']!=ids[p] for p in (A,B)))
        checkpoint('old orders remain rejected after new purchases',all(core.order_row(ids[p])['status']=='rejected' for p in (A,B)))
        checkpoint('new orders can change recipient and sender',latest(A)['recipient_name']=='Zoe Newname' and latest(B)['recipient_name']=='Ece Yeni',str({p:(latest(p)['sender_name'],latest(p)['recipient_name']) for p in (A,B)}))
        checkpoint('new orders reuse delivery addresses when requested',all(latest(p)['address']==core.order_row(ids[p])['address'] for p in (A,B)),str({p:latest(p)['address'] for p in (A,B)}))
        checkpoint('no real channel configured',core.TRANSPORT is None and core.TG is None and is_mock())
    save()
    print('ALL CHECKS PASSED',len(CHECKS),'REPORT',OUT/'report.json',flush=True)

try:
    asyncio.run(main())
except BaseException:
    save()
    raise
