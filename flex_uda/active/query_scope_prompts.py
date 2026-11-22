"""Fixed language-only measurement instructions; no target object labels."""
import json


def term(kind, name, value, part, binding, reference, time, span):
    return dict(kind=kind, name=name, value=value, part=part, binding=binding,
        reference=reference, time=time, negated=False, span=span)


# Synthetic language examples, never annotations about target images.
EXAMPLES = (
    ('person in a yellow coat', dict(subject='person', subject_span='person', logic='and',
        terms=[term('entity','entity','person','none','subject','none','unspecified','person'),
            term('appearance','color','yellow','upper','subject','none','appearance','yellow'),
            term('appearance','garment','coat','upper','subject','none','appearance','coat')], uncertain=[])),
    ('car left of the green van', dict(subject='car', subject_span='car', logic='and',
        terms=[term('entity','entity','car','none','subject','none','unspecified','car'),
            term('spatial','side','left_of','none','subject','other','instant','left of the green van'),
            term('entity','entity','van','none','reference','other','unspecified','van'),
            term('appearance','color','green','whole','reference','other','appearance','green')], uncertain=[])),
)

COMMON = '''Interpret only the supplied referring description. Do not inspect or infer any image, candidate, annotation, dataset identity, or whether a target exists. Extract fallible linguistic structure, not facts about objects.
Return one JSON object, no explanation or Markdown. Required top-level keys: subject, subject_span, logic, terms, uncertain. Required keys in EVERY term: kind, name, value, part, binding, reference, time, negated, span. Do not omit negated or span.
Allowed lowercase values (choose one, never copy the choices):
subject: person, car, vehicle, other, unknown
logic: and, opaque
kind: entity, appearance, spatial, motion, role, activity, opaque
name: entity, color, color_pattern, garment, gender, number, side, team_side, depth, relative_motion, motion, role, activity, pose, other. The name of an entity term is entity; person/car/van belong in value.
part: upper, lower, whole, none, unspecified
binding: subject, reference, unspecified
reference: none, image, ego, team, world, other, unspecified
time: appearance, instant, interval, role_or_action, unspecified
negated: true or false (JSON boolean)
subject_span and every term span: actual nonempty words copied from the supplied description. Never write placeholder words such as exact substring. value: a short canonical interpretation. uncertain: a JSON list of actual ambiguities, or [] if none; never copy a placeholder.
Use an entity term for the subject. Keep every meaningful qualifier; use opaque terms rather than deleting unsupported wording. Flat conjunctions may use and. Unresolved disjunctions, quantifiers, or unclear clause structure use opaque and preserve the entire description. All spans must occur literally in the input; canonical values may normalize synonyms (autos to car, grey to gray).
Bind colors to the noun they modify. A red reference car must not become a red subject. Upper/lower garment colors can be separate terms. A multicolor garment such as blue-and-white shorts is one color_pattern, not two mutually exclusive colors. Do not decide that white person means white clothes; mark ambiguous color scope.
Separate position from motion. The left-side team is a team reference, not necessarily an individual on the image left. Same/opposite direction needs a reference; do not insert ego when unstated. Image-side interpretation of bare left/right is an assumption and must be noted in uncertain with reference unspecified. Left in were left can be a verb; preserve that ambiguity. Do not turn pixel displacement into world motion.
Dancing person alone does not specify whether dancing is a role or an instant action: preserve an activity term with time role_or_action and state the ambiguity. Do not silently erase dancing. Garment/color descriptors can have time appearance; this is a description category, not proof a provided TrackID stays the same person. Keep unclear temporal scope unspecified. An unknown clause is not a false clause.
Do not output probabilities, target identities, ground-truth membership or adaptation recommendations. A syntactically valid structure is not claimed correct. Use opaque if unsure. Keep the object concise, at most twelve terms.
'''

COMMON += '\nComplete examples using synthetic descriptions; follow every field:\n'+ '\n'.join(
    'Description: '+text+'\nJSON: '+json.dumps(answer, separators=(',', ':'))
    for text, answer in EXAMPLES)
COMMON += '\nNow interpret only the new user description. Include every required field with its actual content.\n'

PROMPTS = {
    'PARSE': COMMON+'Read the description and extract its subject, bound predicates, references and temporal scopes. Return the JSON object.',
    'SCOPE_REVIEW': COMMON+'Independently check which noun owns each modifier and what each relation refers to. Preserve uncertainty before returning the same JSON schema.'
}
