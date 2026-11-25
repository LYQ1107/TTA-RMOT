"""Conservative text-only query equivalence and implication constraints.

Human motivation: maintain one interpretation of the same description and
respect 'white-shirt player' implies 'player'. This module uses task text,
not target annotations, learned source accuracy or answers from future videos.
Unparsed content is preserved as an ordered opaque qualifier. Relations are
regularizers under explicitly recorded parsing assumptions, not logical GT.
"""
import re
import numpy as np

COLOURS=r'black|white|gray|grey|silver|red|orange|yellow|green|blue|purple|pink|brown|dark|light'
SPORT=r'basketball|volleyball|football|soccer'
SUBJECT=r'player|person|pedestrian|people|man|woman|car|cars|automobile|automobiles|auto|autos|vehicle|vehicles'


def canonical(text, scene_sport=None):
    text=re.sub(r'[^a-z0-9 -]',' ',text.lower())
    text=re.sub(r'\b(?:dacning|dancing)\b','',text)
    text=re.sub(r'\bt[ -]?shirt\b','tshirt',text)
    text=re.sub(r'\bgrey\b','gray',text)
    text=re.sub(r'\b(?:the|a|an)\b','',text)
    text=' '.join(text.split())
    atoms=set(); consumed=[]
    # Only recognise a subject at the beginning, optionally preceded by colour
    # or sport. A reference object later in the sentence is not the subject.
    match=re.match(r'(?:(?P<colour>'+COLOURS+r')\s+)?(?:(?P<sport>'+SPORT+r')\s+)?(?P<noun>'+SUBJECT+r')\b',text)
    if not match:
        # These named roles entail player but remain distinct from generic player.
        match=re.match(r'(libero|goalkeeper)\b',text)
        if not match:return dict(atoms=(),opaque=text,supported=False)
        atoms.update(('SUBJECT_PERSON','ROLE_PLAYER','ROLE_'+match.group(1)))
    else:
        noun=match.group('noun')
        if noun in ('car','cars','automobile','automobiles','auto','autos'):
            atoms.update(('SUBJECT_VEHICLE','SUBJECT_CAR'))
        elif noun in ('vehicle','vehicles'):atoms.add('SUBJECT_VEHICLE')
        else:
            atoms.add('SUBJECT_PERSON')
            if noun=='player':atoms.add('ROLE_PLAYER')
            if noun in ('man','woman'):atoms.add('GENDER_'+noun)
        if match.group('sport'):
            sport=match.group('sport');sport='football' if sport=='soccer' else sport
            if sport!=scene_sport:atoms.add('SPORT_'+sport)
        if match.group('colour'):
            # "White person" can describe skin, unlike "person in white".
            # Do not silently turn this modifier into a clothing observation.
            prefix='COLOUR_WHOLE_' if noun in ('car','cars','automobile','automobiles','auto','autos','vehicle','vehicles') else 'UNSPECIFIED_COLOUR_'
            atoms.add(prefix+match.group('colour'))
    consumed.append(match.span())
    patterns=[
        (r'\bdressed all in\s+('+COLOURS+r')\b','ALL'),
        (r'\b('+COLOURS+r')\s+(tshirt|shirt|jersey|top|sweater|pants|trousers|skirt|shorts)\b','GARMENT'),
        (r'\b(?:in|wearing)\s+('+COLOURS+r')\s*(?:clothes|clothing|uniform)?\b','WHOLE')]
    occupied=set(i for a,b in consumed for i in range(a,b))
    for pattern,kind in patterns:
        for item in re.finditer(pattern,text):
            if any(i in occupied for i in range(*item.span())):continue
            colour=item.group(1)
            if kind=='ALL':atoms.update(('COLOUR_UPPER_'+colour,'COLOUR_LOWER_'+colour))
            elif kind=='WHOLE':atoms.add('COLOUR_WHOLE_'+colour)
            else:
                garment=item.group(2);part='LOWER' if garment in ('pants','trousers','skirt','shorts') else 'UPPER'
                garment='pants' if garment=='trousers' else garment
                atoms.update(('COLOUR_'+part+'_'+colour,'GARMENT_'+garment))
                if garment=='tshirt':atoms.add('GARMENT_shirt')
            consumed.append(item.span());occupied.update(range(*item.span()))
    remaining=''.join(' ' if i in occupied else c for i,c in enumerate(text))
    # Only grammatical scaffolding is dropped. Direction, motion, reference
    # objects, negation and unrecognised descriptions stay in their original order.
    remaining=re.sub(r'\b(?:with|and|is|are|which|that)\b',' ',remaining)
    opaque=' '.join(remaining.split())
    return dict(atoms=tuple(sorted(atoms)),opaque=opaque,supported=True)


def relations(texts):
    sports=set(re.findall(r'\b('+SPORT+r')\b',' '.join(texts).lower()))
    sports={'football' if s=='soccer' else s for s in sports}
    scene_sport=next(iter(sports)) if len(sports)==1 else None
    parsed=[canonical(text,scene_sport) for text in texts];edges=[];equivalent=[]
    for a,first in enumerate(parsed):
        if not first['supported']:continue
        left=set(first['atoms'])
        for b,second in enumerate(parsed):
            if a==b or not second['supported']:continue
            right=set(second['atoms'])
            # A qualified description entails its unqualified version; a
            # different opaque qualifier is never replaced by a bag-of-words match.
            qualifier_ok=not second['opaque'] or first['opaque']==second['opaque']
            if right<=left and qualifier_ok:
                edges.append((a,b))
                if left==right and first['opaque']==second['opaque'] and a<b:equivalent.append((a,b))
    return parsed,np.asarray(edges,dtype=int).reshape(-1,2),equivalent


def project(probability,edges,*,tolerance=1e-8,max_iterations=10000):
    """Hildreth coordinate solution of min ||p-u||²/2, p[a] <= p[b].

    Queries share the identical per-TrackID probability variable. This is a
    convex consistency projection; correlated descriptions are not multiplied
    as independent observations. No threshold or GT-dependent choice is fitted.
    """
    x=np.array(probability,dtype=float,copy=True);edges=np.asarray(edges,dtype=int).reshape(-1,2)
    assert x.ndim==2 and np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
    if not len(edges):return x,dict(edges=0,iterations=0,converged=True,max_violation=0.,max_change=0.)
    dual=np.zeros((len(edges),x.shape[1]));converged=False
    for iteration in range(1,max_iterations+1):
        largest=0.
        for k,(a,b) in enumerate(edges):
            new=np.maximum(0.,dual[k]+.5*(x[a]-x[b]));change=new-dual[k]
            x[a]-=change;x[b]+=change;dual[k]=new
            largest=max(largest,float(np.max(np.abs(change))))
        violation=float(np.max(x[edges[:,0]]-x[edges[:,1]]))
        if largest<=tolerance and violation<=tolerance:converged=True;break
    assert converged,('Query implication projection did not converge',len(edges),violation,largest)
    # Exact order-cone projection lies in the input probability range. Only
    # floating-point roundoff may leave the unit interval.
    assert x.min()>-1e-7 and x.max()<1+1e-7
    x=np.clip(x,0.,1.)
    return x,dict(edges=len(edges),iterations=iteration,converged=converged,max_violation=max(0.,violation),
        max_change=float(np.max(np.abs(x-probability))),objective=.5*float(np.square(x-probability).sum()))
