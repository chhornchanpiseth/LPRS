class Operation:
    def __init__(self, plate_id):
        self.plate_id = list(plate_id)
        self.char_to_digit = {
            "O": "0",
            "L": "1",
            "Z": "2",
            "A": "4",
            "S": "5",
            "G": "6",
            "T": "7",
            "B": "8",
            "I": "1",
            "E": "6",
            "Q": "0",
            "U": "0",
            "D": "0",
        }
        self.digit_to_char = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "4": "A",
            "5": "S",
            "6": "G",
            "7": "T",
            "8": "B",
        }

    def operation_plate(self):
        try:
            if len(self.plate_id) == 6:
                for i in range(6):
                    if i == 1:
                        if (
                            self.plate_id[i].isdigit()
                            and self.plate_id[i] in self.digit_to_char
                        ):
                            self.plate_id[i] = self.digit_to_char[self.plate_id[i]]
                    else:
                        if (
                            self.plate_id[i].isalpha()
                            and self.plate_id[i] in self.char_to_digit
                        ):
                            self.plate_id[i] = self.char_to_digit[self.plate_id[i]]
            elif len(self.plate_id) == 7:
                for i in range(7):
                    if i == 1 or i == 2:
                        if (
                            self.plate_id[i].isdigit()
                            and self.plate_id[i] in self.digit_to_char
                        ):
                            self.plate_id[i] = self.digit_to_char[self.plate_id[i]]
                    else:
                        if (
                            self.plate_id[i].isalpha()
                            and self.plate_id[i] in self.char_to_digit
                        ):
                            self.plate_id[i] = self.char_to_digit[self.plate_id[i]]
            if self.plate_id and self.plate_id[0] == 4:
                self.plate_id[0] = 1
        except Exception as e:
            print("Error:", e)
        finally:
            return "".join(self.plate_id)
