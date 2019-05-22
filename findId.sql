USE [sbrde]

GO

 

DECLARE      @return_value int

 

EXEC   @return_value = [dbo].[sp_export_dumps]

             @clientId = 127,

             @policyId = 1,

             @from = N'20190422',

             @to = N'20190522'

 

SELECT 'Return Value' = @return_value

 

GO