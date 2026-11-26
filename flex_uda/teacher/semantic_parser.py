"""High-precision deterministic grammar; no GT, no generated pseudo negatives."""
import re
from .teacher_types import Attribute, ParsedExpression

COLORS='black|white|red|blue|green|yellow|purple|brown|pink|gray|grey|orange'
GARMENTS='tshirt|t-shirt|shirt|pants|trousers|skirt|jacket|coat|shorts|dress'


def parse_expression(sentence):
    original=str(sentence);s=' '.join(original.lower().split());notes=[]
    if s.startswith('dacning '):
        s='dancing '+s[len('dacning '):];notes.append('REGISTERED_DACNING_SPELLING_ALIAS_IN_PARSER_ONLY')
    attrs=[]
    if s.startswith('dancing person'):
        attrs.append(Attribute('ACTION','the person is dancing','history'))
        rest=s[len('dancing person'):].strip()
        if rest.startswith('dressed all in '):
            color=rest[len('dressed all in '):]
            if re.fullmatch(COLORS,color):attrs.append(Attribute('APPEARANCE',f'the person is dressed all in {color}','current'))
            else:return ParsedExpression(original,'UNSUPPORTED',(),('UNSUPPORTED_CLOTHING_SCOPE',))
        elif rest.startswith('with '):
            parts=rest[len('with '):].split(' and ')
            for part in parts:
                match=re.fullmatch(rf'({COLORS})\s+({GARMENTS})',part)
                if not match:return ParsedExpression(original,'UNSUPPORTED',(),('UNPARSED_REMAINDER',))
                color,garment=match.groups();attrs.append(Attribute('APPEARANCE',f'the person is wearing {color} {garment}','current'))
        elif rest:return ParsedExpression(original,'UNSUPPORTED',(),('UNPARSED_REMAINDER',))
    else:
        return ParsedExpression(original,'UNSUPPORTED',(),('UNSUPPORTED_EXPRESSION',))
    kinds={a.kind for a in attrs}
    return ParsedExpression(original,'MIXED' if len(kinds)>1 else next(iter(kinds)),tuple(attrs),tuple(notes))
