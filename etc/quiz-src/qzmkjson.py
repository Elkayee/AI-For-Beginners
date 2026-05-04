import json, os
from copy import deepcopy

class QuizMaker:
    def __init__(self, src, dst_dir):
        # Initialize quiz maker
        # Set source and destination
        self.src = src
        self.dst_dir = dst_dir

    def mk_id(self, s):
        # Convert string to ID
        # Parses numeric lesson identifier
        lesson_no = int(s[:-1])
        return lesson_no + (100 if s[-1]=='B' else 200)

    def load_template(self):
        # Load template file
        # Reads JSON structure into dict
        with open('template.json') as f:
            return json.load(f)

    def parse_input(self):
        # Parse input text
        # Returns parsed lessons dictionary
        with open(self.src, encoding='utf-8') as f:
            inp = f.readlines()

        prev_q, prev_l, prev_l_id = None, None, None
        lessons = {}
        for l in inp:
            l = l.strip()
            if l == '': continue
            if l.startswith('+') or l.startswith('-'):
                prev_q['answerOptions'].append({ "answerText" : l[2:], "isCorrect" :  l.startswith('+') })
            elif l.startswith('*'):
                if prev_q:
                    prev_l['quiz'].append(prev_q)
                prev_q = { "questionText" : l[2:], "answerOptions" : [] }
            elif l.startswith('Lesson'):
                if prev_q:
                    prev_l['quiz'].append(prev_q)
                    prev_q = None
                if prev_l is not None:
                    lessons[prev_l_id] = prev_l
                prev_l_id = l[7:l.find(' ',7)]
                prev_l = { "id" : self.mk_id(prev_l_id), "title" : l[l.find(' ',7)+1:], "quiz" : [] }
            else:
                print(f"Error: {l}")

        prev_l['quiz'].append(prev_q)
        lessons[prev_l_id] = prev_l
        return lessons

    def group_lessons(self, lessons, doc):
        # Group parsed lessons
        # Returns merged content dictionary
        lesson_content = {}
        for k, v in lessons.items():
            no = int(k[:-1])
            if no not in lesson_content:
                lesson_content[no] = deepcopy(doc)
            lesson_content[no][0]['quizzes'].append(v)
        return lesson_content

    def write_index(self, lesson_content):
        # Write JavaScript index
        # Generates ES6 module exports
        with open(os.path.join(self.dst_dir, 'index.js'), 'w', encoding='utf-8') as f:
            for i, k in enumerate(lesson_content.keys()):
                f.write(f'import x{k} from "./lesson-{k}.json";\n')
            t = ', '.join([f"{i} : x{k}[0]" for i, k in enumerate(lesson_content.keys())])
            f.write(f"const quiz = {{ {t} }}; \n")
            f.write("export default quiz;")

    def write_lessons(self, lesson_content):
        # Write lesson JSONs
        # Saves individual JSON files
        for k, v in lesson_content.items():
            with open(os.path.join(self.dst_dir, f"lesson-{k}.json"), 'w', encoding='utf-8') as f:
                json.dump(v, f, indent=2)

    def run(self):
        # Execute parsing process
        # Generates output artifacts
        doc = self.load_template()
        lessons = self.parse_input()
        lesson_content = self.group_lessons(lessons, doc)
        self.write_index(lesson_content)
        self.write_lessons(lesson_content)

if __name__ == '__main__':
    qm = QuizMaker('questions-en.txt', '../quiz-app/src/assets/translations/en')
    qm.run()
