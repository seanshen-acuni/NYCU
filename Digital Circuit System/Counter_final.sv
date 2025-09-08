module Counter(
	// Input signals
	clk,
	rst_n,
	// Output signals
	clk2
);
input  clk, rst_n;
output logic clk2;
reg     [1:0]temp;
always @ (posedge clk or negedge rst_n) begin  
	if (rst_n == 0) begin 
		clk2  <=  1'b0;
		temp  <= 2'b00;  
	end
        else begin
	case (temp)
		2'b00:  begin
			temp <= 2'b01;
			clk2 <=  1'b1;
			end
		2'b01:  begin
			temp <= 2'b10;
                        clk2 <=  1'b1;
			end
		2'b10:  begin
			temp <= 2'b11;
                        clk2 <=  1'b0;
			end
		2'b11:  begin
			temp <= 2'b00;
                        clk2 <=  1'b0;
			end
	endcase
        end
end
endmodule
